import Component from '@glimmer/component';
import { DEBUG } from '@glimmer/env';

type Args = {
  // no args
};

export default class Debug extends Component<Args> {
  IS_DEVELOPMENT = DEBUG;

  constructor(owner: unknown, args: Args) {
    super(owner, args);

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

    const css = '/dist/css/main.min.css';
    const js = '/dist/js/app.min.js';

    await installFile('https://code.jquery.com/jquery-3.5.1.slim.min.js');
    await installFile(css, 'style');
    await installFile(js);
  }
}

async function installFile(url: string, type = 'script') {
  // fetching the URL instead of directly loading in a script
  // tag allows us to get around CORS issues
  const response = await fetch(url);
  const script = await response.text();

  const element = document.createElement(type);

  element.innerHTML = script;

  document.body.appendChild(element);
}
