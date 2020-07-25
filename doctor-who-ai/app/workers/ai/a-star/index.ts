import random from 'random';

import { InternalMove, ALL_MOVES, MOVE_KEY_MAP } from '../consts';
import { clone } from '../rnn/utils';
import { imitateMove } from '../game';

type SearchNode = {
  value: {
    model: Game2048;
  };
  children: SearchNode[];
  move?: number;
  parent?: SearchNode;
  weightedScore?: number;
};

export function guidedMove(numActions: number, gameManager: Game2048) {
  let result: InternalMove[] = [];

  let bestNode: any;
  let bestScore = 0;
  let bestHops = 1000;

  let rootNode: SearchNode = {
    value: { model: gameManager.serialize() },
    children: [],
  };

  function updateBest(childNode: SearchNode) {
    if (childNode === rootNode) {
      return;
    }

    if (childNode.weightedScore < bestScore) {
      return;
    }

    // if the score is equal, let's choose the least hops
    let root = childNode;
    let hops = 0;

    while (root.parent !== undefined && root.parent.move !== undefined) {
      root = root.parent;
      hops++;
    }

    if (childNode.weightedScore > bestScore) {
      bestNode = root;
      bestScore = childNode.weightedScore;
    }
  }

  function expandTree(node: SearchNode, level: number) {
    updateBest(node);

    if (level >= 4) {
      return;
    }

    const enumerateMoves = () => {
      for (let move of ALL_MOVES) {
        let copyOfModel = clone(node.value);
        let moveData = imitateMove(copyOfModel.model, move);

        if (!moveData.wasMoved) {
          continue;
        }

        let scoreChange = moveData.score - gameManager.score;
        let weightedScore = level === 0 ? scoreChange : scoreChange / level;

        node.children.push({
          // penalize scores with higher depth
          weightedScore,

          value: moveData,
          children: [],
          move: MOVE_KEY_MAP[move],
          parent: node,
        });
      }
    };

    enumerateMoves();

    for (let childNode of node.children) {
      expandTree(childNode, level + 1);
    }
  }

  expandTree(rootNode, 0);

  if (bestNode && bestNode.move) {
    result.push(bestNode.move);
  }

  // [0, numActions]
  let generateMove = () => random.int(0, numActions - 1);

  while (result.length < 4) {
    let move = generateMove();

    if (!result.includes(move)) {
      result.push(move);
    }
  }

  return { sorted: result };
}
