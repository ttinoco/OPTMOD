/** @file node.c
 * 
 * This file is part of OPTMOD
 *
 * Copyright (c) 2019, Tomas Tinoco De Rubira. 
 *
 * OPTMOD is released under the BSD 2-clause license.
 */

#include "node.h"

struct Node {

  long id;
  NodeType type;

};

Node* NODE_array_new(int num) {
  int i;
  Node* n = (Node*)malloc(sizeof(Node)*num);
  for (i = 0; i < num; i++) {
    n->id = 0;
    n->type = NODE_TYPE_UNKNOWN;
  }
  return n;
}
