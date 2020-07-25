# doctor-who-ai

This README outlines the details of collaborating on this Ember application.
A short introduction of this app could easily go here.

## Prerequisites

You will need the following things properly installed on your computer.

* [Git](https://git-scm.com/)
* [Node.js](https://nodejs.org/)
* [Yarn](https://yarnpkg.com/)
* [Ember CLI](https://ember-cli.com/)
* [Google Chrome](https://google.com/chrome/)

## Installation

* `git clone <repository-url>` this repository
* `cd doctor-who-ai`
* `yarn install`

## Running / Development

* `ember serve`
* Visit your app at [http://localhost:4200](http://localhost:4200).
* Visit your tests at [http://localhost:4200/tests](http://localhost:4200/tests).

### WebWorkers

 - would like to use html-next/skyrocket
 - waiting on a few issues to be resolved https://github.com/html-next/skyrocket/issues
   - including typescript support
 - packages added to support custom web-worker pipeline
   - promise-worker-bi
   - @babel/preset-env
   - @babel/preset-typescript
   - @babel/plugin-proposal-class-properties
   - @babel/plugin-proposal-object-rest-spread
   - rollup
   - rollup-plugin-filesize
   - rollup-plugin-terser
   - @rollup/plugin-node-resolve
   - @rollup/plugin-babel
   - rollup-plugin-commonjs
   - broccoli-rollup

### Running Tests

* `ember test`
* `ember test --server`

### Linting

* `yarn lint:hbs`
* `yarn lint:js`
* `yarn lint:js --fix`

### Building

* `ember build` (development)
* `ember build --environment production` (production)

### Deploying

Specify what it takes to deploy your app.

## Further Reading / Useful Links

* [ember.js](https://emberjs.com/)
* [ember-cli](https://ember-cli.com/)
* Development Browser Extensions
  * [ember inspector for chrome](https://chrome.google.com/webstore/detail/ember-inspector/bmdblncegkenkacieihfhpjfppoconhi)
  * [ember inspector for firefox](https://addons.mozilla.org/en-US/firefox/addon/ember-inspector/)
* General Deep Learning
  * [Reinforcement Learning in the Browser](https://medium.com/@pierrerouhard/reinforcement-learning-in-the-browser-an-introduction-to-tensorflow-js-9a02b143c099)
  * [Reinforcement Deep Q Learning for Playing a Game in Unity](https://medium.com/ml2vec/reinforcement-deep-q-learning-for-playing-a-game-in-unity-d2577fb50a81)
  * [Crash Course in Deep Q Networks](https://towardsdatascience.com/qrash-course-deep-q-networks-from-the-ground-up-1bbda41d3677)
  * [Open A.I. Baslines DQN](https://openai.com/blog/openai-baselines-dqn/)
