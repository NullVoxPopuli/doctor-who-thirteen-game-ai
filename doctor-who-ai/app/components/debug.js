import Component from '@glimmer/component';
import { DEBUG } from '@glimmer/env';

export default class Debug extends Component {
  IS_DEVELOPMENT = DEBUG;

  constructor() {
    super(...arguments);

    if (DEBUG) {
      setupLocalGame();
    }
  }
}

async function setupLocalGame() {
  if (DEBUG) {
    let css = 'https://raw.githubusercontent.com/NullVoxPopuli/doctor-who-thirteen-game-ai/master/game-backup/dist/css/main.min.css';
    let js = 'https://raw.githubusercontent.com/NullVoxPopuli/doctor-who-thirteen-game-ai/master/game-backup/dist/js/app.min.css';

    async function installFile(url, type = 'script') {
      // fetching the URL instead of directly loading in a script
      // tag allows us to get around CORS issues
      let response = await fetch(url);
      let script = await response.text();

      let element = document.createElement(type);

      element.innerHTML = script;

      document.body.appendChild(element);
    }

    await installFile(css);
    await installFile(js);
  }
}
