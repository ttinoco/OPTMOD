/** @file node.h
 * 
 * This file is part of OPTMOD
 *
 * Copyright (c) 2019, Tomas Tinoco De Rubira. 
 *
 * OPTMOD is released under the BSD 2-clause license.
 */

#include <stdlib.h>

typedef struct Node Node;

typedef enum {
  NODE_TYPE_UNKNOWN,
  NODE_TYPE_CONSTANT,
  NODE_TYPE_VARIABLE,
  NODE_TYPE_ADD,
  NODE_TYPE_SUBTRACT,
  NODE_TYPE_NEGATE,
  NODE_TYPE_MULTIPLY,
  NODE_TYPE_SIN,
  NODE_TYPE_COS
} NodeType;


Node* NODE_array_new(int num);
