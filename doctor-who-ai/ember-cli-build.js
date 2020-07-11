'use strict';

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
  app.trees.vendor = new UnwatchedDir('public');
  app.trees.dist = new UnwatchedDir('public');

  return mergeTrees([app.toTree()]);
};
