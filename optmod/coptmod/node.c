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
  NodeType type;
  long id;
  double value;
  Node* arg1;
  Node* arg2;
  Node** args;
  int num_args;
  UT_hash_handle hh;
};

Node* NODE_hash_add(Node* hash, Node* n) {
  HASH_ADD(hh,hash,id,sizeof(long),n);
  return hash;
}

Node* NODE_hash_find(Node* hash, long id) {
  Node* n;
  HASH_FIND(hh,hash,&id,sizeof(long),n);
  return n;
}

void NODE_hash_del(Node* hash) {
  while (hash != NULL)
    HASH_DELETE(hh,hash,hash);
}

Node* NODE_array_new(int num) {
  int i;
  Node* n = (Node*)malloc(sizeof(Node)*num);
  for (i = 0; i < num; i++)
    NODE_init(n+i);
  return n;
}

Node* NODE_array_get(Node* n, int i) {
  if (n)
    return n+i;
  else
    return NULL;
}

void NODE_array_del(Node* n, int num) {
  
}

void NODE_init(Node* n) {
  if (n) {
    n->type = NODE_TYPE_UNKNOWN;
    n->id = 0;
    n->value = 0;
    n->arg1 = NULL;
    n->arg2 = NULL;
    n->args = NULL;
    n->num_args = 0;
  }
}

void NODE_set_type(Node* n, NodeType type) {
  if (n)
    n->type = type;
}

void NODE_set_id(Node* n, long id) {
  if (n)
    n->id = id;
}

void NODE_set_value(Node* n, double value) {
  if (n)
    n->value = value;
}

void NODE_set_arg1(Node* n, Node* arg1) {
  if (n)
    n->arg1 = arg1;
}

void NODE_set_arg2(Node* n, Node* arg2) {
  if (n)
    n->arg2 = arg2;
}

void NODE_set_args(Node* n, Node** args, int num) {
  if (n) {
    if (n->args)
      free(args);
    n->args = args;
    n->num_args = num;
  }
}

