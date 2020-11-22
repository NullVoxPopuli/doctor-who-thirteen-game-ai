import random from 'random';

import { InternalMove, ALL_MOVES, MOVE_KEY_MAP, DirectionKey } from '../consts';
import { clone, isNewGame, isEqual, emptySpaces } from '../rnn/utils';
import { imitateMove } from '../game';
import { MoveResult } from '../types';

type SearchNode = {
  value: {
    model: Game2048;
  };
  children: SearchNode[];
  move?: InternalMove;
  keyMove?: DirectionKey;
  parent?: SearchNode;
  weightedScore?: number;
};

const MAX_DEPTH = 8;

let tree: SearchNode;

/**
 * Build out an A* search tree up to MAX_DEPTH.
 * Maintain that depth without recomputing the in between games.
 * Calculate the score of each node via getting the max score within
 * the weighting based on depth.
 */
export function getMove(game: Game2048): MoveResult {
  if (isNewGame(game)) {
    tree = {
      value: {
        model: game,
      },
      children: [],
    };
  }

  console.time('A* getMove');

  let result = getMoveFromTree(game);

  console.timeEnd('A* getMove');

  if (!result) {
    throw new Error('why no move available?');
  }

  return { move: result };
}

function rootGame() {
  return tree.value.model;
}

function isTreeCurrent(game: Game2048) {
  return isEqual(game.grid.cells, tree.value.model.grid.cells);
}

function trim(game: Game2048) {
  if (isTreeCurrent(game)) return;
  if (!tree.children || tree.children.length === 0) return;

  for (let subTree of tree.children) {
    if (isEqual(subTree.value.model.grid.cells, game.grid.cells)) {
      tree = subTree;

      return;
    }
  }

  throw new Error(`Tree could not correctly be trimmed.`);
}

function getMoveFromTree(game: Game2048) {
  trim(game);
  appendChildren();

  let bestChild = findBest(tree);

  return bestChild.keyMove;
}

function findBest(node: SearchNode | SearchNode[]): SearchNode {
  if (Array.isArray(node)) {
    return node.reduce(function (prev, current) {
      return findBest(prev).value.model.score > findBest(current).value.model.score
        ? prev
        : current;
    }, node[0]);
  }

  if (!node.children || node.children.length === 0) {
    return node;
  }

  return findBest(node.children);
}

function enumerateMoves(node: SearchNode, level = 0): void {
  for (let move of ALL_MOVES) {
    let copyOfModel = clone(node.value);
    let moveData = imitateMove(copyOfModel.model, move, true);

    if (!moveData.wasMoved) {
      continue;
    }

    node.children.push({
      value: moveData,
      children: [],
      move: MOVE_KEY_MAP[move],
      keyMove: move,
      parent: node,
    });
  }
}

function appendChildren(root = tree, level = 0) {
  if (root.children.length === 0) {
    return expandTree(root, level);
  }

  for (let child of root.children) {
    appendChildren(child, level + 1);
  }
}

function expandTree(node: SearchNode, level: number): void {
  if (level === MAX_DEPTH) {
    return;
  }

  // TODO: this may need to change to remove the new random space,
  //       rather than hacking at the GameManager to just not add it
  enumerateMoves(node, level);

  let withNewMoves = [];

  for (let childNode of node.children) {
    let empties = emptySpaces(childNode.value.model);

    // if (empties.length <= 8) {
      // because we can't predict where the new tile is going to end up...
      for (let emptySpace of emptySpaces(childNode.value.model)) {
        let { row, column } = emptySpace;

        let gameWithMove = clone(childNode.value.model);

        gameWithMove.grid.cells[column][row] = {
          value: 2,
          position: { x: column, y: row },
        };

        withNewMoves.push({
          ...childNode,
          value: {
            model: gameWithMove,
          },
        });
      }
    // }
  }

  if (withNewMoves.length > 0) {
    node.children = withNewMoves;
  }

  for (let childNode of node.children) {
    expandTree(childNode, level + 1);
  }
}
