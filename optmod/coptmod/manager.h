/** @file manager.h
 * 
 * This file is part of OPTMOD
 *
 * Copyright (c) 2019, Tomas Tinoco De Rubira. 
 *
 * OPTMOD is released under the BSD 2-clause license.
 */

#include "node.h"

typedef struct Manager Manager;

void MANAGER_add_node(Manager* m, int type, long id, double value, long* arg_ids, int num_args);
void MANAGER_del(Manager* m);
int MANAGER_get_max_nodes(Manager* m);
int MANAGER_get_num_nodes(Manager* m);
int MANAGER_get_num_inputs(Manager* m);
int MANAGER_get_num_outputs(Manager* m);
Manager* MANAGER_new(int num_inputs, int num_outputs);
void MANAGER_set_output_node(Manager* m, int index, long id);
void MANAGER_set_input_var(Manager* m, int index, long id);
void MANAGER_show(Manager* m);
