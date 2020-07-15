/* global Chartist */
import Component from '@glimmer/component';
import { inject as service } from '@ember/service';

export default class Graph extends Component {
  @service history;

  get chartData() {
    return {
      series: [this.history.scores, this.history.averageScores],
    };
  }

  get options() {
    return {
      plugins: [
        Chartist.plugins.legend({
          legendNames: ['Score', 'Average Score'],
        }),
      ],
      fullWidth: true,
      axisY: {
        onlyInteger: true,
      },
      chartPadding: {
        left: 40,
      },
    };
  }
}
