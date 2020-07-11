import { helper } from '@ember/component/helper';

import { round } from 'doctor-who-ai/utils';

export default helper(function ([num] /*, hash*/) {
  return round(num);
});
