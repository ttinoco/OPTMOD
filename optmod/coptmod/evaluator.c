/** @file evaluator.c
 * 
 * This file is part of OPTMOD
 *
 * Copyright (c) 2019, Tomas Tinoco De Rubira. 
 *
 * OPTMOD is released under the BSD 2-clause license.
 */

#include "evaluator.h"

struct Evaluator {

  int max_nodes;
  int num_nodes;
  Node* nodes;
  
  int num_inputs;
  Node** inputs;
  
  int num_outputs;
  double* outputs;
  
  Node* hash;
};

int EVALUATOR_get_max_nodes(Evaluator* e) {
  if (e)
    return e->max_nodes;
  else
    return 0;
}

int EVALUATOR_get_num_nodes(Evaluator* e) {
  if (e)
    return e->num_nodes;
  else
    return 0;
}

int EVALUATOR_get_num_inputs(Evaluator* e) {
  if (e)
    return e->num_inputs;
  else
    return 0;
}

int EVALUATOR_get_num_outputs(Evaluator* e) {
  if (e)
    return e->num_outputs;
  else
    return 0;
}

Evaluator* EVALUATOR_new(int num_inputs, int num_outputs) {
  int i;
  Evaluator* e = (Evaluator*)malloc(sizeof(Evaluator));  
  e->max_nodes = num_outputs;
  e->num_nodes = 0;
  e->nodes = NODE_array_new(e->max_nodes);
  e->num_inputs = num_inputs;
  e->num_outputs = num_outputs;
  e->inputs = (Node**)malloc(sizeof(Node*)*num_inputs);
  e->outputs = (double*)malloc(sizeof(double)*num_outputs);
  e->hash = NULL;
  for (i = 0; i < e->num_inputs; i++)
    e->inputs[i] = NULL;
  for (i = 0; i < e->num_outputs; i++)
    e->outputs[i] = 0.;
  return e;
}

void EVALUATOR_inc_num_nodes(Evaluator* e) {

  // Local vars
  int i;
  Node* n;
  Node* new_n;
  Node* new_nodes;
  Node* new_hash;
  int new_max_nodes;

  // Check
  if (!e)
    return;

  // Increment
  e->num_nodes += 1;

  // Dynamic resize
  if (e->num_nodes >= e->max_nodes) {

    // New nodes
    new_max_nodes = 2*e->max_nodes;
    new_nodes = NODE_array_new(new_max_nodes);

    // New hash
    new_hash = NULL;
    for (i = 0; i < e->num_nodes; i++) {
      n = NODE_array_get(e->nodes, i);
      new_n = NODE_array_get(new_nodes, i);
      NODE_set_id(new_n, NODE_get_id(n));
      new_hash = NODE_hash_add(new_hash, new_n);
    }

    // Copy old node data
    for (i = 0; i < e->num_nodes; i++) {
      n = NODE_array_get(e->nodes, i);
      new_n = NODE_array_get(new_nodes, i);
      NODE_copy_from_node(new_n, n, new_hash);
    }

    // Update hash
    NODE_hash_del(e->hash);
    e->hash = new_hash;

    // Update inputs
    for (i = 0; i < e->num_inputs; i++)
      e->inputs[i] = NODE_hash_find(e->hash, NODE_get_id(e->inputs[i]));

    // Update nodes
    NODE_array_del(e->nodes, e->num_nodes);
    e->nodes = new_nodes;
    e->max_nodes = new_max_nodes;
  }
}

void EVALUATOR_add_node(Evaluator* e, int type, long id, double value, long* arg_ids, int num_args) {

  int i;
  Node* n;
  Node* arg;
  Node** args;
  
  if (!e)
    return;
  
  // Root
  n = NODE_hash_find(e->hash, id);  
  if (!n) {
    n = NODE_array_get(e->nodes, e->num_nodes);
    NODE_set_id(n, id);
    e->hash = NODE_hash_add(e->hash, n);
    EVALUATOR_inc_num_nodes(e);
  }
  NODE_set_type(n, type);
  NODE_set_value(n, value);
  
  // args
  args = (Node**)malloc(sizeof(Node*)*num_args);
  for (i = 0; i < num_args; i++) {
    arg = NODE_hash_find(e->hash, arg_ids[i]);
    if (!arg) {
      arg = NODE_array_get(e->nodes, e->num_nodes);
      NODE_set_id(arg, arg_ids[i]);
      e->hash = NODE_hash_add(e->hash, arg);
      EVALUATOR_inc_num_nodes(e);
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

void EVALUATOR_del(Evaluator* e) {
  if (e) {
    NODE_hash_del(e->hash);
    NODE_array_del(e->nodes, e->num_nodes);
    free(e->inputs);
    free(e->outputs);
    free(e);
  }
}

void EVALUATOR_set_output_node(Evaluator* e, int index, long id) {

  Node* n;
  
  if (!e)
    return;

  n = NODE_hash_find(e->hash, id);
  if (0 <= index && 0 < e->num_outputs)
    NODE_set_output_index(n, index);
}

void EVALUATOR_set_input_var(Evaluator* e, int index, long id) {
  
  Node* n;

  if (!e)
    return;

  n = NODE_hash_find(e->hash, id);
  if (0 <= index && index < e->num_inputs && NODE_get_type(n) == NODE_TYPE_VARIABLE)
    e->inputs[index] = n;
}

void EVALUATOR_show(Evaluator* e) {

  int i;

  if (!e)
    return;

  printf("\n");
  printf("Evaluator\n");
  printf("num_inputs: %d\n", e->num_inputs);
  printf("num_outputs: %d\n", e->num_outputs);
  printf("max_nodes: %d\n", e->max_nodes);
  printf("num_nodes: %d\n\n", e->num_nodes);

  printf("inputs:\n");
  for (i = 0; i < e->num_inputs; i++)
    printf("%ld, ", NODE_get_id(e->inputs[i]));
  printf("\n\n");

  printf("outputs:\n");
  for (i = 0; i < e->num_outputs; i++)
    printf("%.2e, ", e->outputs[i]);
  printf("\n\n");
  
  printf("nodes:\n\n");
  for (i = 0; i < e->num_nodes; i++) {
    NODE_show(NODE_array_get(e->nodes, i));
    printf("\n");
  }
}
