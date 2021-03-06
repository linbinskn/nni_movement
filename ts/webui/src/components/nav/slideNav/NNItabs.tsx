import * as React from 'react';
import { NavLink } from 'react-router-dom';

import { getPrefix } from '@static/function';

const OVERVIEWTABS = (
    <NavLink to='/oview' activeClassName='selected' className='common-tabs'>
        Overview
    </NavLink>
);

const DETAILTABS = (
    <NavLink to='/detail' activeClassName='selected' className='common-tabs'>
        Trials detail
    </NavLink>
);

const NNILOGO = (
    <NavLink to='/oview'>
        <img src={(getPrefix() || '') + '/logo.png'} alt='NNI logo' style={{ height: 40 }} />
    </NavLink>
);

export { OVERVIEWTABS, DETAILTABS, NNILOGO };
