// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import assert from 'assert';
import { PlacementConstraint } from 'common/trainingService';
import { getLogger, Logger } from 'common/log';
import { randomSelect } from 'common/utils';
import { GPUInfo, ScheduleResultType } from '../common/gpuData';
import { EnvironmentInformation } from './environment';
import { RemoteMachineEnvironmentInformation } from './remote/remoteConfig';
import { TrialDetail } from './trial';

type SCHEDULE_POLICY_NAME = 'random' | 'round-robin' | 'recently-idle';

export class GpuSchedulerSetting {
    public useActiveGpu: boolean = false;
    public maxTrialNumberPerGpu: number = 1;
}

export type GpuScheduleResult = {
    resultType: ScheduleResultType;
    environment: EnvironmentInformation | undefined;
    gpuIndices: GPUInfo[] | undefined;
};

/**
 * A simple GPU scheduler implementation
 */
export class GpuScheduler {

    // private readonly machineExecutorMap: Set<TrialDetail>;
    private readonly log: Logger = getLogger('GpuScheduler');
    private readonly policyName: SCHEDULE_POLICY_NAME = 'recently-idle';
    private defaultSetting: GpuSchedulerSetting;
    private roundRobinIndex: number = 0;

    /**
     * Constructor
     * @param environments map from remote machine to executor
     */
    constructor(gpuSchedulerSetting: GpuSchedulerSetting | undefined = undefined) {
        if (undefined === gpuSchedulerSetting) {
            gpuSchedulerSetting = new GpuSchedulerSetting();
        }
        this.defaultSetting = gpuSchedulerSetting;
    }

    public setSettings(gpuSchedulerSetting: GpuSchedulerSetting): void {
        this.defaultSetting = gpuSchedulerSetting;
    }

    /**
     * Schedule a machine according to the constraints (requiredGPUNum)
     * @param defaultRequiredGPUNum the default required GPU number when constraint.type === 'None'
     */
    public scheduleMachine(environments: EnvironmentInformation[], constraint: PlacementConstraint,
        defaultRequiredGPUNum: number | undefined, trialDetail: TrialDetail): GpuScheduleResult {
        if (constraint.type == 'None' || constraint.type == 'GPUNumber') {
            let requiredGPUNum = 0;
            if (constraint.type == 'None') {
                if (defaultRequiredGPUNum === undefined) {
                    requiredGPUNum = 0;
                } else {
                    requiredGPUNum = defaultRequiredGPUNum;
                }
            } else if (constraint.type == 'GPUNumber') {
                const gpus = constraint.gpus as Array<number>;
                // TODO: remove the following constraint when supporting distributed trial
                if (gpus.length != 1) {
                    throw new Error("Placement constraint of GPUNumber must have exactly one number.");
                }
                requiredGPUNum = gpus[0];
            }

            assert(requiredGPUNum >= 0);
            // Step 1: Check if required GPU number not exceeds the total GPU number in all machines
            const eligibleEnvironments: EnvironmentInformation[] = environments.filter((environment: EnvironmentInformation) =>
                environment.defaultGpuSummary === undefined || requiredGPUNum === 0 ||
                (requiredGPUNum !== undefined && environment.defaultGpuSummary.gpuCount >= requiredGPUNum));
            if (eligibleEnvironments.length === 0) {
                // If the required gpu number exceeds the upper limit of all machine's GPU number
                // Return REQUIRE_EXCEED_TOTAL directly
                return ({
                    resultType: ScheduleResultType.REQUIRE_EXCEED_TOTAL,
                    gpuIndices: undefined,
                    environment: undefined,
                });
            }

            // Step 2: Allocate Host/GPU for specified trial job
            // Currenty the requireGPUNum parameter for all trial jobs are identical.
            if (requiredGPUNum > 0) {
                // Trial job requires GPU
                const result: GpuScheduleResult | undefined = this.scheduleGPUHost(environments, requiredGPUNum, trialDetail);
                if (result !== undefined) {
                    return result;
                }
            } else {
                // Trail job does not need GPU
                const allocatedRm: EnvironmentInformation = this.selectMachine(environments, environments);

                return this.allocateHost(requiredGPUNum, allocatedRm, [], trialDetail);
            }

            return {
                resultType: ScheduleResultType.TMP_NO_AVAILABLE_GPU,
                gpuIndices: undefined,
                environment: undefined,
            };
        } else {
            assert(constraint.type === 'Device')
            if (constraint.gpus.length == 0) {
                throw new Error("Device constraint is used but no device is specified.");
            }
            const gpus = constraint.gpus as Array<[string, number]>;
            const selectedHost = gpus[0][0];

            const differentHosts: Array<[string, number]> = gpus.filter((gpuTuple: [string, number]) => gpuTuple[0] != selectedHost);
            if (differentHosts.length >= 1) {
                //TODO: remove this constraint when supporting multi-host placement
                throw new Error("Device constraint does not support using multiple hosts")
            }
            if (environments.length == 0) {
                return {
                    resultType: ScheduleResultType.TMP_NO_AVAILABLE_GPU,
                    gpuIndices: undefined,
                    environment: undefined,
                };
            }
            for (const environment of environments) {
                if(!('rmMachineMeta' in environment)){
                    //TODO: remove this constraint when supporting other training services
                    throw new Error(`Environment Device placement constraint only supports remote training service for now.`);
                }
            }
            //TODO: 
            const eligibleEnvironments: EnvironmentInformation[] = environments.filter(
                (environment: EnvironmentInformation) =>
                    (environment as RemoteMachineEnvironmentInformation).rmMachineMeta != undefined &&
                    (environment as RemoteMachineEnvironmentInformation).rmMachineMeta?.host == selectedHost);
            if (eligibleEnvironments.length === 0) {
                throw new Error(`The the required host (host: ${selectedHost}) is not found.`);
            }
            const selectedEnvironment = eligibleEnvironments[0];
            const availableResources = this.gpuResourceDetection([selectedEnvironment]);
            const selectedGPUs: Array<GPUInfo> = [];

            if (selectedEnvironment.defaultGpuSummary === undefined) {
                //GPU summary may not be ready, retry until it is ready
                return {
                    resultType: ScheduleResultType.TMP_NO_AVAILABLE_GPU,
                    gpuIndices: undefined,
                    environment: undefined,
                };
            }
            for (const gpuTuple of gpus) {
                const gpuIdx: number = gpuTuple[1];
                if (gpuIdx >= selectedEnvironment.defaultGpuSummary.gpuCount) {
                    throw new Error(`The gpuIdx of placement constraint ${gpuIdx} exceeds gpuCount of the host ${selectedHost}`);
                }

                if (availableResources.has(selectedEnvironment)) {
                    for (const gpuInfo of availableResources.get(selectedEnvironment)!) {
                        if (gpuInfo.index === gpuIdx) {
                            selectedGPUs.push(gpuInfo);
                        }
                    }
                }
            }
            if (selectedGPUs.length === constraint.gpus.length) {
                for (const gpuInfo of selectedGPUs) {
                    let num = selectedEnvironment.defaultGpuSummary?.assignedGpuIndexMap.get(gpuInfo.index);
                    if (num === undefined) {
                        num = 0;
                    }
                    selectedEnvironment.defaultGpuSummary?.assignedGpuIndexMap.set(gpuInfo.index, num + 1);
                }
                return {
                    resultType: ScheduleResultType.SUCCEED,
                    environment: selectedEnvironment,
                    gpuIndices: selectedGPUs,
                };
            } else {
                return {
                    resultType: ScheduleResultType.TMP_NO_AVAILABLE_GPU,
                    gpuIndices: undefined,
                    environment: undefined,
                };
            }
        }
    }

    /**
     * remove the job's gpu reversion
     */
    public removeGpuReservation(trial: TrialDetail): void {
        if (trial.environment !== undefined &&
            trial.environment.defaultGpuSummary !== undefined &&
            trial.assignedGpus !== undefined &&
            trial.assignedGpus.length > 0) {

            for (const gpuInfo of trial.assignedGpus) {
                const defaultGpuSummary = trial.environment.defaultGpuSummary;
                const num: number | undefined = defaultGpuSummary.assignedGpuIndexMap.get(gpuInfo.index);
                if (num !== undefined) {
                    if (num === 1) {
                        defaultGpuSummary.assignedGpuIndexMap.delete(gpuInfo.index);
                    } else {
                        defaultGpuSummary.assignedGpuIndexMap.set(gpuInfo.index, num - 1);
                    }
                }
            }
        }
    }

    private scheduleGPUHost(environments: EnvironmentInformation[], requiredGPUNumber: number, trial: TrialDetail): GpuScheduleResult | undefined {
        const totalResourceMap: Map<EnvironmentInformation, GPUInfo[]> = this.gpuResourceDetection(environments);
        const qualifiedEnvironments: EnvironmentInformation[] = [];
        totalResourceMap.forEach((gpuInfos: GPUInfo[], environment: EnvironmentInformation) => {
            if (gpuInfos !== undefined && gpuInfos.length >= requiredGPUNumber) {
                qualifiedEnvironments.push(environment);
            }
        });
        if (qualifiedEnvironments.length > 0) {
            const allocatedEnvironment: EnvironmentInformation = this.selectMachine(qualifiedEnvironments, environments);
            const gpuInfos: GPUInfo[] | undefined = totalResourceMap.get(allocatedEnvironment);
            if (gpuInfos !== undefined) { // should always true
                return this.allocateHost(requiredGPUNumber, allocatedEnvironment, gpuInfos, trial);
            } else {
                assert(false, 'gpuInfos is undefined');
            }
        }
        return undefined;
    }

    /**
     * Detect available GPU resource for an environment
     * @returns Available GPUs on environments
     */
    private gpuResourceDetection(environments: EnvironmentInformation[]): Map<EnvironmentInformation, GPUInfo[]> {
        const totalResourceMap: Map<EnvironmentInformation, GPUInfo[]> = new Map<EnvironmentInformation, GPUInfo[]>();
        environments.forEach((environment: EnvironmentInformation) => {
            // Assgin totoal GPU count as init available GPU number
            if (environment.defaultGpuSummary !== undefined) {
                const defaultGpuSummary = environment.defaultGpuSummary;
                const availableGPUs: GPUInfo[] = [];
                const designatedGpuIndices: Set<number> = new Set<number>(environment.usableGpus);
                if (designatedGpuIndices.size > 0) {
                    for (const gpuIndex of designatedGpuIndices) {
                        if (gpuIndex >= environment.defaultGpuSummary.gpuCount) {
                            throw new Error(`Specified GPU index not found: ${gpuIndex}`);
                        }
                    }
                }

                if (undefined !== defaultGpuSummary.gpuInfos) {
                    defaultGpuSummary.gpuInfos.forEach((gpuInfo: GPUInfo) => {
                        // if the GPU has active process, OR be reserved by a job,
                        // or index not in gpuIndices configuration in machineList,
                        // or trial number on a GPU reach max number,
                        // We should NOT allocate this GPU
                        // if users set useActiveGpu, use the gpu whether there is another activeProcess
                        if (designatedGpuIndices.size === 0 || designatedGpuIndices.has(gpuInfo.index)) {
                            if (defaultGpuSummary.assignedGpuIndexMap !== undefined) {
                                const num: number | undefined = defaultGpuSummary.assignedGpuIndexMap.get(gpuInfo.index);
                                const maxTrialNumberPerGpu: number = environment.maxTrialNumberPerGpu ? environment.maxTrialNumberPerGpu : this.defaultSetting.maxTrialNumberPerGpu;
                                const useActiveGpu: boolean = environment.useActiveGpu ? environment.useActiveGpu : this.defaultSetting.useActiveGpu;
                                if ((num === undefined && (!useActiveGpu && gpuInfo.activeProcessNum === 0 || useActiveGpu)) ||
                                    (num !== undefined && num < maxTrialNumberPerGpu)) {
                                    availableGPUs.push(gpuInfo);
                                }
                            } else {
                                throw new Error(`occupiedGpuIndexMap is undefined!`);
                            }
                        }
                    });
                }
                totalResourceMap.set(environment, availableGPUs);
            }
        });

        return totalResourceMap;
    }

    private selectMachine(qualifiedEnvironments: EnvironmentInformation[], allEnvironments: EnvironmentInformation[]): EnvironmentInformation {
        assert(qualifiedEnvironments !== undefined && qualifiedEnvironments.length > 0);

        if (this.policyName === 'random') {
            return randomSelect(qualifiedEnvironments);
        } else if (this.policyName === 'round-robin') {
            return this.roundRobinSelect(qualifiedEnvironments, allEnvironments);
        } else if (this.policyName === 'recently-idle') {
            return this.recentlyIdleSelect(qualifiedEnvironments, allEnvironments);
        } else {
            throw new Error(`Unsupported schedule policy: ${this.policyName}`);
        }
    }

    // Select the environment which is idle most recently. If all environments are not idle, use round robin to select an environment.
    private recentlyIdleSelect(qualifiedEnvironments: EnvironmentInformation[], allEnvironments: EnvironmentInformation[]): EnvironmentInformation {
        const now = Date.now();
        let selectedEnvironment: EnvironmentInformation | undefined = undefined;
        let minTimeInterval = Number.MAX_SAFE_INTEGER;
        for (const environment of qualifiedEnvironments) {
            if (environment.latestTrialReleasedTime > 0 && (now - environment.latestTrialReleasedTime) < minTimeInterval) {
                selectedEnvironment = environment;
                minTimeInterval = now - environment.latestTrialReleasedTime;
            }
        }
        if (selectedEnvironment === undefined) {
            return this.roundRobinSelect(qualifiedEnvironments, allEnvironments);
        }
        selectedEnvironment.latestTrialReleasedTime = -1;
        return selectedEnvironment;
    }

    private roundRobinSelect(qualifiedEnvironments: EnvironmentInformation[], allEnvironments: EnvironmentInformation[]): EnvironmentInformation {
        while (!qualifiedEnvironments.includes(allEnvironments[this.roundRobinIndex % allEnvironments.length])) {
            this.roundRobinIndex++;
        }

        return allEnvironments[this.roundRobinIndex++ % allEnvironments.length];
    }

    private selectGPUsForTrial(gpuInfos: GPUInfo[], requiredGPUNum: number): GPUInfo[] {
        // Sequentially allocate GPUs
        return gpuInfos.slice(0, requiredGPUNum);
    }

    private allocateHost(requiredGPUNum: number, environment: EnvironmentInformation,
        gpuInfos: GPUInfo[], trialDetails: TrialDetail): GpuScheduleResult {
        assert(gpuInfos.length >= requiredGPUNum);
        const allocatedGPUs: GPUInfo[] = this.selectGPUsForTrial(gpuInfos, requiredGPUNum);
        const defaultGpuSummary = environment.defaultGpuSummary;
        if (undefined === defaultGpuSummary) {
            throw new Error(`Environment ${environment.id} defaultGpuSummary shouldn't be undefined!`);
        }

        allocatedGPUs.forEach((gpuInfo: GPUInfo) => {
            let num: number | undefined = defaultGpuSummary.assignedGpuIndexMap.get(gpuInfo.index);
            if (num === undefined) {
                num = 0;
            }
            defaultGpuSummary.assignedGpuIndexMap.set(gpuInfo.index, num + 1);
        });
        trialDetails.assignedGpus = allocatedGPUs;

        return {
            resultType: ScheduleResultType.SUCCEED,
            environment: environment,
            gpuIndices: allocatedGPUs,
        };
    }
}
