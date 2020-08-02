'use strict';

// ---------------------------------------------------------------
// ---- web-worker building code
// ---------------------------------------------------------------
const Rollup = require('broccoli-rollup');

const commonjs = require('@rollup/plugin-commonjs');
// import babel, { getBabelOutputPlugin } from '@rollup/plugin-babel';
const { babel, getBabelOutputPlugin } = require('@rollup/plugin-babel');
const resolve = require('@rollup/plugin-node-resolve');
const { terser } = require('rollup-plugin-terser');
const filesize = require('rollup-plugin-filesize');
const alias = require('@rollup/plugin-alias');

const AssetRev = require('broccoli-asset-rev');

const path = require('path');
const fs = require('fs');
let cwd = process.cwd();
let workerRoot = path.join(cwd, 'app', 'workers');
let extensions = ['.js', '.ts'];

function detectWorkers() {
  let workers = {};
  let dir = fs.readdirSync(workerRoot);

  for (let i = 0; i < dir.length; i++) {
    let name = dir[i];

    workers[name] = path.join(workerRoot, name, 'index.ts');
  }

  return workers;
}

function configureWorkerTree({ isProduction, hash }) {
  return ([name, entryPath]) => {
    let workerDir = path.join(workerRoot, name);

    let rollupTree = new Rollup(workerDir, {
      rollup: {
        input: entryPath,
        // watch true prevents editors from safely saving
        watch: false,
        output: [
          {
            file: `workers/${name}.js`,
            format: 'cjs',
            // plugins: [
            //   getBabelOutputPlugin({
            //     presets: [
            //       require('@babel/preset-env'),
            //       {
            //         // targets: '> 2%, not IE 11, not dead',
            //         // corejs: {
            //         //   version: 3,
            //         // },
            //       },
            //     ],
            //   }),
            // ],
          },
        ],
        plugins: [
          alias({
            entries: [
              { find: 'consts', replacement: path.resolve(workerDir, 'consts') },
              { find: name, replacement: path.resolve(workerDir) },

            ],
          }),
          resolve({
            extensions,
            browser: true,
            preferBuiltins: true,
          }),
          commonjs({
            // include: ['node_modules/**'],
          }),
          babel({
            extensions,
            babelHelpers: 'bundled',
            presets: [
              [
                require('@babel/preset-env'),
                {
                  useBuiltIns: 'usage',
                  targets: '> 2%, not IE 11, not dead',
                  corejs: {
                    version: 3,
                  },
                },
              ],
              require('@babel/preset-typescript'),
            ],
            plugins: [
              [
                require('@babel/plugin-transform-typescript'),
                { allowDeclareFields: true, allowNamespaces: true },
              ],
              require('@babel/plugin-proposal-class-properties'),
              require('@babel/plugin-proposal-object-rest-spread'),
            ],
            exclude: /node_modules/,
          }),
          ...(isProduction ? [terser()] : []),
          filesize(),
        ],
      },
    });

    if (!isProduction) {
      return rollupTree;
    }

    return new AssetRev(rollupTree, {
      customHash: hash,
    });
  };
}

function buildWorkerTrees(env) {
  let inputs = detectWorkers();
  let workerBuilder = configureWorkerTree(env);
  let workerTrees = Object.entries(inputs).map(workerBuilder);

  return workerTrees;
}

// ---------------------------------------------------------------
// ---- normal ember-cli-babel.js
// ---------------------------------------------------------------

const EmberApp = require('ember-cli/lib/broccoli/ember-app');
const gitRev = require('git-rev-sync');
const mergeTrees = require('broccoli-merge-trees');
const UnwatchedDir = require('broccoli-source').UnwatchedDir;

const { EMBROIDER, CONCAT_STATS } = process.env;

module.exports = function (defaults) {
  let environment = EmberApp.env();
  let isProduction = environment === 'production';

  let version = gitRev.short();

  let env = {
    isProduction,
    isTest: environment === 'test',
    version,
    CONCAT_STATS,
    EMBROIDER,
  };

  console.log(env);

  let app = new EmberApp(defaults, {
    hinting: false,
    autoprefixer: {
      enabled: false,
      sourcemaps: false,
    },
    sourcemaps: {
      enabled: false,
    },
    fingerprint: {
      // need stable URL for bookmarklet to load
      enabled: false,
    },
  });

  app.trees.public = new UnwatchedDir('public');
  // app.trees.vendor = new UnwatchedDir('public');
  // app.trees.dist = new UnwatchedDir('public');

  app.import('node_modules/chartist-plugin-legend/chartist-plugin-legend.js');

  return mergeTrees([app.toTree(), ...buildWorkerTrees(env)]);
};
