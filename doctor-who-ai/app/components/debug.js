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
    // let iframe = document.createElement('iframe');

    // iframe.src = '/game-backup/index.html';

    // document.body.appendChild(iframe);

    let css = '/dist/css/main.min.css';
    let js = '/dist/js/app.min.js';

    await installFile('https://code.jquery.com/jquery-3.5.1.slim.min.js');
    await installFile(css, 'style');
    await installFile(js);
  }
}

async function installFile(url, type = 'script') {
  // fetching the URL instead of directly loading in a script
  // tag allows us to get around CORS issues
  let response = await fetch(url);
  let script = await response.text();

  let element = document.createElement(type);

  element.innerHTML = script;

  document.body.appendChild(element);
}
