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
  int index;
  int type;
  char type_name[NODE_BUFFER_SIZE];
  long id;
  double value;
  Node* arg1;
  Node* arg2;
  Node** args;
  int num_args;
  UT_hash_handle hh;
};

void NODE_copy_from_node(Node* n, Node* other, Node* hash) {

  int i;
  Node** args;

  if (!n || !other)
    return;

  n->type = other->type;
  strcpy(n->type_name, other->type_name);
  n->id = other->id;
  n->value = other->value;
  if (other->arg1)
    n->arg1 = NODE_hash_find(hash, other->arg1->id);
  if (other->arg2)
    n->arg2 = NODE_hash_find(hash, other->arg2->id);
  if (other->num_args > 0) {
    args = (Node**)malloc(sizeof(Node*)*other->num_args);
    for (i = 0; i < other->num_args; i++)
      args[i] = NODE_hash_find(hash, other->args[i]->id);
    NODE_set_args(n, args, other->num_args);
  }
}

long NODE_get_id(Node* n) {
  if (n)
    return n->id;
  else
    return 0;
}

int NODE_get_type(Node* n) {
  if (n)
    return n->type;
  else
    return NODE_TYPE_UNKNOWN;
}


char* NODE_get_type_name(Node* n) {
  if (n) {
    switch (n->type) {
    case NODE_TYPE_UNKNOWN:
      strcpy(n->type_name, "unknown");
      break;
    case NODE_TYPE_CONSTANT:
      strcpy(n->type_name, "constant");
      break;
    case NODE_TYPE_VARIABLE:
      strcpy(n->type_name, "variable");
      break;
    case NODE_TYPE_ADD:
      strcpy(n->type_name, "add");
      break;
    case NODE_TYPE_SUBTRACT:
      strcpy(n->type_name, "subtract");
      break;
    case NODE_TYPE_NEGATE:
      strcpy(n->type_name, "negate");
      break;
    case NODE_TYPE_MULTIPLY:
      strcpy(n->type_name, "multiply");
      break;
    case NODE_TYPE_SIN:
      strcpy(n->type_name, "sin");
      break;
    case NODE_TYPE_COS:
      strcpy(n->type_name, "cos");
      break;
    default:
      strcpy(n->type_name, "error");
    }
    return n->type_name;
  }
  else
    return NULL;
}

double NODE_get_value(Node* n) {

  int i;
  double temp;

  if (!n)
    return 0;
  
  switch (n->type) {
    
  case NODE_TYPE_UNKNOWN:
    return 0;
  case NODE_TYPE_CONSTANT:
    return n->value;
  case NODE_TYPE_VARIABLE:
    return n->value;
  case NODE_TYPE_ADD:
    if (n->arg1 && n->arg2)
      return NODE_get_value(n->arg1) + NODE_get_value(n->arg2);
    temp = 0;
    for (i = 0; i < n->num_args; i++)
      temp += NODE_get_value(*(n->args+i));
    return temp;
  case NODE_TYPE_SUBTRACT:
    return NODE_get_value(n->arg1) - NODE_get_value(n->arg2);
  case NODE_TYPE_NEGATE:
    return -NODE_get_value(n->arg1);
  case NODE_TYPE_MULTIPLY:
    return NODE_get_value(n->arg1)*NODE_get_value(n->arg2);
  case NODE_TYPE_SIN:
    return sin(NODE_get_value(n->arg1));
  case NODE_TYPE_COS:
    return cos(NODE_get_value(n->arg1));
  default:
    return 0;
  }
}

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
  for (i = 0; i < num; i++) {
    NODE_init(n+i);
    (n+i)->index = i;
  }
  return n;
}

int NODE_get_index(Node* n) {
  if (n)
    return n->index;
  else
    return -1;
}

Node* NODE_array_get(Node* n, int i) {
  if (n)
    return n+i;
  else
    return NULL;
}

void NODE_array_del(Node* n, int num) {
  int i;
  for (i = 0; i < num; i++) {
    if ((n+i)->args && (n+i)->num_args > 0)
      free((n+i)->args);
  }
  free(n);
}

void NODE_init(Node* n) {
  if (n) {
    n->index = 0;
    n->type = NODE_TYPE_UNKNOWN;
    strcpy(n->type_name, "");
    n->id = 0;
    n->value = 0;
    n->arg1 = NULL;
    n->arg2 = NULL;
    n->args = NULL;
    n->num_args = 0;
  }
}

void NODE_set_type(Node* n, int type) {
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
      free(n->args);
    n->args = args;
    n->num_args = num;
  }
}

void NODE_show(Node* n) {
  int i;
  if (n) {
    printf("Node\n");
    printf("type: %s\n", NODE_get_type_name(n));
    printf("id: %ld\n", n->id);
    printf("value: %.4e\n", n->value);
    printf("arg1 id: %ld\n", NODE_get_id(n->arg1));
    printf("arg2 id: %ld\n", NODE_get_id(n->arg2));
    printf("num args: %d\n", n->num_args );
    for (i = 0; i < n->num_args; i++)
      printf("%ld ", NODE_get_id(n->args[i]));
    printf("\n");
  }
}
