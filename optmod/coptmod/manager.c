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
  
};

Manager* MANAGER_new(int max_nodes) {
  Manager* m = (Manager*)malloc(sizeof(Manager));
  m->max_nodes = max_nodes;
  m->num_nodes = 0;
  m->nodes = NODE_array_new(max_nodes);
  return m;
}

void MANAGER_del(Manager* m) {
  if (m)
    free(m);
}
