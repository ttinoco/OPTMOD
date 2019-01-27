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

int MANAGER_get_max_nodes(Manager* m) {
  if (m)
    return m->max_nodes;
  else
    return 0;
}

int MANAGER_get_num_nodes(Manager* m) {
  if (m)
    return m->num_nodes;
  else
    return 0;
}

Manager* MANAGER_new(int max_nodes) {
  Manager* m = (Manager*)malloc(sizeof(Manager));
  m->max_nodes = max_nodes;
  m->num_nodes = 0;
  m->nodes = NODE_array_new(max_nodes);
  m->hash = NULL;
  return m;
}

void MANAGER_inc_num_nodes(Manager* m) {
  
  int i;
  Node* n;
  Node* new_n;
  Node* new_nodes;
  Node* new_hash;
  int new_max_nodes;
  
  if (!m)
    return;

  m->num_nodes += 1;
  
  if (m->num_nodes >= m->max_nodes) {
    new_hash = NULL;
    new_max_nodes = 2*m->max_nodes;
    new_nodes = NODE_array_new(new_max_nodes);
    for (i = 0; i < m->num_nodes; i++) {
      n = NODE_array_get(m->nodes, i);
      new_n = NODE_array_get(new_nodes, i);
      NODE_set_id(new_n, NODE_get_id(n));
      new_hash = NODE_hash_add(new_hash, new_n);
    }
    for (i = 0; i < m->num_nodes; i++) {
      n = NODE_array_get(m->nodes, i);
      new_n = NODE_array_get(new_nodes, i);
      NODE_copy_from_node(new_n, n, new_hash);
    }
    NODE_hash_del(m->hash);
    NODE_array_del(m->nodes, m->num_nodes);
    m->hash = new_hash;
    m->nodes = new_nodes;
    m->max_nodes = new_max_nodes;
  }
}

void MANAGER_add_node(Manager* m, int type, long id, double value, long* arg_ids, int num_args) {

  int i;
  Node* n;
  Node* arg;
  Node** args;
  
  if (!m)
    return;
  
  // Root
  n = NODE_hash_find(m->hash, id);  
  if (!n) {
    n = NODE_array_get(m->nodes, m->num_nodes);
    NODE_set_id(n, id);
    m->hash = NODE_hash_add(m->hash, n);
    MANAGER_inc_num_nodes(m);
  }
  NODE_set_type(n, type);
  NODE_set_value(n, value);
  
  // args
  args = (Node**)malloc(sizeof(Node*)*num_args);
  for (i = 0; i < num_args; i++) {
    arg = NODE_hash_find(m->hash, arg_ids[i]);
    if (!arg) {
      arg = NODE_array_get(m->nodes, m->num_nodes);
      NODE_set_id(arg, arg_ids[i]);
      m->hash = NODE_hash_add(m->hash, arg);
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
    NODE_array_del(m->nodes, m->num_nodes);
    free(m);
  }
}

void MANAGER_show(Manager* m) {

  int i;

  if (!m)
    return;

  printf("\n");
  printf("Manager\n");
  printf("max_nodes: %d\n", m->max_nodes);
  printf("num_nodes: %d\n", m->num_nodes);
  printf("nodes:\n\n");

  for (i = 0; i < m->num_nodes; i++) {
    NODE_show(NODE_array_get(m->nodes, i));
    printf("\n");
  }
}
