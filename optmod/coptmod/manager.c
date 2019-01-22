/** @file manager.c
 * 
 * This file is part of OPTMOD
 *
 * Copyright (c) 2019, Tomas Tinoco De Rubira. 
 *
 * OPTMOD is released under the BSD 2-clause license.
 */

#include "manager.h"

struct Manager {

  int max_nodes;
  int num_nodes;
  
  Node* nodes;
  Node* hash;
};

Manager* MANAGER_new(int max_nodes) {
  Manager* m = (Manager*)malloc(sizeof(Manager));
  m->max_nodes = max_nodes;
  m->num_nodes = 0;
  m->nodes = NODE_array_new(max_nodes);
  m->hash = NULL;
  return m;
}

void MANAGER_inc_num_nodes(Manager* m) {

  // increment number of nodes
  // dynamically reallcate larger array and updates pointers and hash tables
}

void MANAGER_add_node(Manager* m, NodeType type, long id, double value, long* arg_ids, int num_args) {

  int i;
  Node* n;
  Node* arg;
  Node** args;
  
  if (!m)
    return;

  // Root
  n = NODE_hash_find(m->nodes, id);
  if (!n) {
    n = NODE_array_get(m->nodes, m->num_nodes);
    NODE_set_id(n, id);
    NODE_hash_add(m->hash, n);
    MANAGER_inc_num_nodes(m);
  }
  NODE_set_type(n, type);
  NODE_set_value(n, value);
  
  // args
  args = (Node**)malloc(sizeof(Node*)*num_args);
  for (i = 0; i < num_args; i++) {
    arg = NODE_hash_find(m->nodes, arg_ids[i]);
    if (!arg) {
      arg = NODE_array_get(m->nodes, m->num_nodes);
      NODE_set_id(arg, arg_ids[i]);
      NODE_hash_add(m->hash, arg);
      MANAGER_inc_num_nodes(m);
    }
    args[i] = arg;
  }

  if (num_args <= 2) {
    NODE_set_arg1(n, args[0]);
    if (num_args > 1)
      NODE_set_arg2(n, args[1]);
    free(args);
  }
  else
    NODE_set_args(n, args, num_args);
}

void MANAGER_del(Manager* m) {
  if (m) {
    NODE_hash_del(m->hash);
    NODE_array_del(m->nodes, m->max_nodes);
    free(m);
  }
}
