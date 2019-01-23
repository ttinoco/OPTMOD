/** @file node.h
 * 
 * This file is part of OPTMOD
 *
 * Copyright (c) 2019, Tomas Tinoco De Rubira. 
 *
 * OPTMOD is released under the BSD 2-clause license.
 */

#ifndef __NODE_HEADER__
#define __NODE_HEADER__

#include <stdio.h>
#include <stdlib.h>
#include "uthash.h"

#define NODE_BUFFER_SIZE 100

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
Node* NODE_array_get(Node* n, int i);
void NODE_array_del(Node* n, int num);
Node* NODE_hash_add(Node* hash, Node* n);
Node* NODE_hash_find(Node* hash, long id);
void NODE_hash_del(Node* hash);
void NODE_init(Node* n);
void NODE_set_type(Node* n, NodeType type);
void NODE_set_id(Node* n, long id);
void NODE_set_value(Node* n, double value);
void NODE_set_arg1(Node* n, Node* arg1);
void NODE_set_arg2(Node* n, Node* arg2);
void NODE_set_args(Node* n, Node** args, int num);
void NODE_show(Node* n);

#endif
