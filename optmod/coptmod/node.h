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
#include <math.h>
#include "uthash.h"

#define NODE_BUFFER_SIZE 100

#define NODE_TYPE_UNKNOWN 0
#define NODE_TYPE_CONSTANT 1
#define NODE_TYPE_VARIABLE 2
#define NODE_TYPE_ADD 3
#define NODE_TYPE_SUBTRACT 4
#define NODE_TYPE_NEGATE 5
#define NODE_TYPE_MULTIPLY 6
#define NODE_TYPE_SIN 7
#define NODE_TYPE_COS 8

typedef struct Node Node;

Node* NODE_array_new(int num);
Node* NODE_array_get(Node* n, int i);
void NODE_array_del(Node* n, int num);
void NODE_copy_from_node(Node* n, Node* other, Node* hash);
int NODE_get_index(Node* n);
long NODE_get_id(Node* n);
int NODE_get_type(Node* n);
double NODE_get_value(Node* n);
char* NODE_get_type_name(Node* n);
Node* NODE_hash_add(Node* hash, Node* n);
Node* NODE_hash_find(Node* hash, long id);
void NODE_hash_del(Node* hash);
void NODE_init(Node* n);
void NODE_set_type(Node* n, int type);
void NODE_set_id(Node* n, long id);
void NODE_set_value(Node* n, double value);
void NODE_set_arg1(Node* n, Node* arg1);
void NODE_set_arg2(Node* n, Node* arg2);
void NODE_set_args(Node* n, Node** args, int num);
void NODE_show(Node* n);

#endif
