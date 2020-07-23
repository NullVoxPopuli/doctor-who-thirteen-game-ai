import Route from '@ember/routing/route';

export default class ApplicationRoute extends Route {
  beforeModel() {
    document.body.classList.add('grid');
    document.body.classList.add('grid-col');
  }
}
