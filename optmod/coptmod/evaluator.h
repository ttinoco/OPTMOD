/** @file evaluator.h
 * 
 * This file is part of OPTMOD
 *
 * Copyright (c) 2019, Tomas Tinoco De Rubira. 
 *
 * OPTMOD is released under the BSD 2-clause license.
 */

#include "node.h"

typedef struct Evaluator Evaluator;

void EVALUATOR_add_node(Evaluator* e, int type, long id, double value, long* arg_ids, int num_args);
void EVALUATOR_del(Evaluator* e);
void EVALUATOR_eval(Evaluator* e, double* var_values);
int EVALUATOR_get_max_nodes(Evaluator* e);
int EVALUATOR_get_num_nodes(Evaluator* e);
int EVALUATOR_get_num_inputs(Evaluator* e);
int EVALUATOR_get_num_outputs(Evaluator* e);
double* EVALUATOR_get_values(Evaluator* e);
Evaluator* EVALUATOR_new(int num_inputs, int num_outputs);
void EVALUATOR_set_output_node(Evaluator* e, int index, long id);
void EVALUATOR_set_input_var(Evaluator* e, int index, long id);
void EVALUATOR_show(Evaluator* e);
