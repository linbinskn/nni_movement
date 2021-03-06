// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import { TrialConfig } from 'training_service/common/trialConfig';
import { EnvironmentInformation } from '../environment';
import { AMLClient } from '../aml/amlClient';

export class AMLClusterConfig {
    public readonly subscriptionId: string;
    public readonly resourceGroup: string;
    public readonly workspaceName: string;
    public readonly computeTarget: string;
    public maxTrialNumPerGpu?: number;

    constructor(subscriptionId: string, resourceGroup: string, workspaceName: string, computeTarget: string,
                maxTrialNumPerGpu?: number) {
        this.subscriptionId = subscriptionId;
        this.resourceGroup = resourceGroup;
        this.workspaceName = workspaceName;
        this.computeTarget = computeTarget;
        this.maxTrialNumPerGpu = maxTrialNumPerGpu;
    }
}

export class AMLTrialConfig extends TrialConfig {
    public readonly image: string;
    public readonly command: string;
    public readonly codeDir: string;

    constructor(codeDir: string, command: string, image: string) {
        super("", codeDir, 0);
        this.codeDir = codeDir;
        this.command = command;
        this.image = image;
    }
}

export class AMLEnvironmentInformation extends EnvironmentInformation {
    public amlClient?: AMLClient;
    public currentMessageIndex: number = -1;
}
