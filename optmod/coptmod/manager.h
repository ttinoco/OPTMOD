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

Manager* MANAGER_new(int max_nodes);
void MANAGER_del(Manager* m);
