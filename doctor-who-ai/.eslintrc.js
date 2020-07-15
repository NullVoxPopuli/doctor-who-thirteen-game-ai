'use strict';

const prettier = {
  rules: {
    'prettier/prettier': [
      'error',
      {
        semi: true,
        trailingComma: 'es5',
        tabWidth: 2,
        singleQuote: true,
        printWidth: 100,
      },
    ],
  },
};

const base = {
  parser: '@typescript-eslint/parser',
  parserOptions: {},
  plugins: ['ember', 'prettier', 'decorator-position'],
  extends: [
    'eslint:recommended',
    'plugin:ember/recommended',
    'plugin:decorator-position/ember',
    'prettier',
  ],
  rules: {
    'prefer-const': 'off', // const has misleading safety implications
    'getter-return': ['error', { allowImplicit: true }],
    'no-console': [
      'error',
      { allow: ['debug', 'warn', 'error', 'info', 'group', 'groupEnd', 'groupCollapsed'] },
    ],

    'padding-line-between-statements': [
      'error',
      { blankLine: 'always', prev: '*', next: 'return' },
      { blankLine: 'always', prev: '*', next: 'block-like' },
      { blankLine: 'always', prev: 'block-like', next: '*' },
      { blankLine: 'always', prev: ['const', 'let'], next: '*' },
      { blankLine: 'any', prev: ['const', 'let'], next: ['const', 'let'] },
    ],

    ...prettier.rules,
  },
};

const typescript = {
  plugins: ['@typescript-eslint'],
  extends: ['plugin:@typescript-eslint/recommended', 'prettier', 'prettier/@typescript-eslint'],
  rules: {
    'no-unused-vars': 'off', // doesn't understand type imports

    // @typescript-eslint
    '@typescript-eslint/no-use-before-define': 'off', // not applicable due to how the runtime is
    '@typescript-eslint/prefer-optional-chain': 'error', // much concise

    // prefer inference
    '@typescript-eslint/explicit-function-return-type': 'off',

    ...prettier.rules,
  },
};

module.exports = {
  root: true,
  ...base,
  env: {
    es6: true,
    browser: true,
    node: false,
  },
  overrides: [
    {
      files: ['app/**/*.ts', 'types/**/*.d.ts'],
      ...typescript,
    },
    {
      files: ['workers/**/*.js', 'workers/**/*.ts'],
      env: {
        worker: true,
      },
    },
    // node files
    {
      files: [
        '.eslintrc.js',
        '.template-lintrc.js',
        'ember-cli-build.js',
        'testem.js',
        'blueprints/*/index.js',
        'config/**/*.js',
        'lib/*/index.js',
        'server/**/*.js',
      ],
      parserOptions: {
        sourceType: 'script',
      },
      env: {
        browser: false,
        node: true,
      },
      plugins: ['node'],
      extends: ['plugin:node/recommended'],
      rules: {
        // this can be removed once the following is fixed
        // https://github.com/mysticatea/eslint-plugin-node/issues/77
        'node/no-unpublished-require': 'off',
        'no-console': 'off',
      },
    },
  ],
};
