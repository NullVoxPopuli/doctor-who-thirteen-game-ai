// NOTE: decorators do not exist in browsers, so we can't
//       use any sort of fancy auto-"bind" decoration :(
// poor man's Dependency Injection
const container = {
  ui: undefined,
  ai: undefined,
};

const VALUE_MAP = {
  /* eslint-disable prettier/prettier */
  2:     1, 4:      2, 8:      3, 16:    4,
  32:    5, 64:     6, 128:    7, 256:   8,
  512:   9, 1024:  10, 2048:  11, 4096: 12,
  8192: 13, 16384: 14, 32768: 15,
  /* eslint-enable prettier/prettier */
};

const DOCTOR_NUMBER_MAP = {
  1: '01 - William Hartnell',
  2: '02 - Patrick Troughton',
  3: '03 - Jon Pertwee',
  4: '04 - Tom Baker',
  5: '05 - Peter Davison',
  6: '06 - Colin Baker',
  7: '07 - Sylvester McCoy',
  8: '08 - Paul McGann',
  9: 'War - John Hurt',
  10: '09 - Christopher Eccleston',
  11: '10 - David Tennant',
  12: '11 - Matt Smith',
  13: '12 - Peter Capaldi',
  14: '13 - Jodie Whittaker',
};

function biggestTile(game) {
  let tiles = game.grid.cells
    .map((row) => row.map((cell) => (cell ? cell.value : 1)))
    .flat();

  let value = Math.max(...tiles);

  return { value, num: VALUE_MAP[value] };
}

function round(num) {
  return Math.round(num * 100) / 100;
}

class AIWorker {
  static async create() {
    let ai = new AIWorker();

    container.ai = ai;
    await container.ai.setup();

    return ai;
  }

  setup = async () => {
    // fetching the URL instead of directly loading in a script
    // tag allows us to get around CORS issues
    let workerUrl =
      'https://raw.githubusercontent.com/NullVoxPopuli/doctor-who-thirteen-game-ai/master/worker.js';

    let response = await fetch(workerUrl);
    let script = await response.text();
    let blob = new Blob([script], { type: 'text/javascript' });

    this.worker = new Worker(URL.createObjectURL(blob));
    this.worker.onmessage = this.onMessage;
  };

  send = (data) => {
    this.worker.postMessage(data);
  };

  onMessage = (e) => {
    let { data } = e;

    switch (data.type) {
      case 'ack':
      case 'ready':
        console.debug(`Received: ${JSON.stringify(data)}`);

        return;
      case 'move':
        if (!data.move) {
          console.error(`No move was generated`, data);

          return;
        }

        if (data.trainingData) {
          localStorage.setItem('training', JSON.stringify(data.trainingData));
        }

        return container.ui.keyDown(data.move);
      default:
        console.error(data);
        throw new Error('Unrecognized Message');
    }
  };

  requestNextMove = (game, algorithm) => {
    if (!this.startTime) {
      this.startTime = new Date();
    }

    let biggest = biggestTile(game).num;

    let totalTime = new Date() - this.startTime;

    container.ui.updateStatus({
      topDoctor: DOCTOR_NUMBER_MAP[biggest],
      totalTime,
    });

    let trainingData;

    if (algorithm === 'RNN') {
      trainingData = JSON.parse(localStorage.getItem('training'));
    }

    this.send({
      type: 'run',
      game,
      algorithm,
      trainingData,
    });
  };
}

class UIComponent {
  constructor({ onRun, enableAutoRetry }) {
    this.args = { onRun, enableAutoRetry };

    this.mount();
    this.update();
  }

  mount = () => {
    let mountPoint = document.createComment('div');

    mountPoint.id = 'ai-mount-point';

    document.body.prepend(mountPoint);

    this.render();

    document
      .querySelector('.ai-container .ai-buttons button')
      .addEventListener('click', this.args.onRun);

    document
      .querySelector('.ai-container .ai-buttons input')
      .addEventListener('click', this.args.enableAutoRetry);
  };

  update = (data) => {
    this.data = data;

    this.mountPoint.innerHTML = this.render();
  };

  render = () => {
    return `
      <style>
        .ai-container {
          display: grid; grid-gap: 0.5rem;
          position: fixed; top: 0.5rem; left: 0.5rem;
          background: white; color: black;
          padding: 0.5rem;
          box-shadow: 2px 2px 2px rgba(0,0,0,0.5);
          border-radius: 0.25rem;
          font-size: 0.75rem;
        }
        .ai-buttons {
          display: grid;
          grid-gap: 0.5rem;
          grid-auto-flow: column;
        }

        .container {
          margin-top: 8rem;
        }

        .ai-container dt {
          font-weight: bold;
        }
      </style>

      <div class="ai-container">
        <div class="ai-buttons">
          <button type='button'>
            Run A.I. (RNN)
          </button>

          <label>
            Auto-Retry
            <input type='checkbox'>
          </label>
        </div>

        <p class='ai-stats'>
          This Session,<br>
          <dl>
            <dt>Total Games</dt> <dd>${this.data.numGames}</dd>
            <dt>Average Score</dt> <dd>${this.data.averageScore}</dd>
            <dt>Best Score</dt> <dd>${this.data.bestScore}</dd>
            <dt>Average Game Length</dt> <dd>${this.data.averageGameLength} minutes</dd>
            <dt>Current Top Doctor<dt> <dd>${this.data.topDoctor}</dd>
          </dl>
          <br>
        </p>
      </div>
    `;
  };
}

class UI {
  static async create() {
    let ui = new UI();

    container.ui = ui;
    container.ui.setup();

    return ui;
  }

  gameHistory = [];

  setup = () => {
    this.component = new UIComponent({
      onRun: () => this.runAI('RNN'),
      toggleAutoRetry: (e) => {
        this.isAutoRetryEnabled = e.target.checked;

        this.autoRetry();
      },
    });
  };

  updateStats = () => {
    let scores = this.gameHistory.map((h) => h.score);
    let times = this.gameHistory.map((h) => h.totalTime);
    let bestScore = Math.max(...scores);
    let averageTime = times.reduce((a, b) => a + b, 0) / times.length;
    let averageScore = round(scores.reduce((a, b) => a + b, 0) / scores.length);
    let averageGameLength = round(averageTime / 1000 / 60);

    this.component.update({
      numGames: scores.length,
      averageScore: averageScore || 0,
      bestScore: bestScore || 0,
      averageGameLength: averageGameLength || 0,
      topDoctor: this.topDoctor,
    });
  };

  updateStatus = ({ topDoctor, totalTime }) => {
    this.topDoctor = topDoctor;
    this.totalTime = totalTime;

    this.updateStats();
  };

  get isGameOver() {
    return Boolean(document.querySelector('.game-over'));
  }

  autoRetry = () => {
    if (!this.isAutoRetryEnabled) {
      return;
    }

    if (this.isGameOver) {
      let score = parseInt(
        document.querySelector('.score-container').textContent,
        10
      );

      this.gameHistory.push({ score, totalTime: this.totalTime });

      container.ai.startTime = undefined;

      document.querySelector('.retry-button').click();

      setTimeout(() => this.requestNextMove(), 1000);
    }

    // check every 10 seconds
    setTimeout(() => this.autoRetry(), 10000);
  };

  runAI = (algorithm) => {
    this.algorithm = algorithm;

    this.requestNextMove();
  };

  requestNextMove = () => {
    let model = JSON.parse(localStorage.getItem('gameState'));

    if (model !== null && !model.over) {
      container.ai.requestNextMove(model, this.algorithm);
    }
  };

  keyDown = (k) => {
    let oEvent = document.createEvent('KeyboardEvent');

    function defineConstantGetter(name, value) {
      Object.defineProperty(oEvent, name, {
        get() {
          return value;
        },
      });
    }

    defineConstantGetter('keyCode', k);
    defineConstantGetter('which', k);
    defineConstantGetter('metaKey', false);
    defineConstantGetter('shiftKey', false);
    defineConstantGetter('target', { tagName: '' });

    /* eslint-disable */
    oEvent.initKeyboardEvent('keydown',
      true, true, document.defaultView, false, false, false, false, k, k
    );
    /* eslint-enable */

    document.dispatchEvent(oEvent);

    setTimeout(() => {
      this.requestNextMove();
    }, 100);
  };
}

async function boot() {
  await AIWorker.create();
  await UI.create();

  container.ai.send({ type: 'ready' });
}

boot();
