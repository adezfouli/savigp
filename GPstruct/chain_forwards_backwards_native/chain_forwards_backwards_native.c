/*
Forward-backward algorithm to compute logZ and the log-likelihood of chain CRFs. Restriction on CRF factor graph: defined from unary factors and binary (edge) factors.

Author: Sebastien Bratieres, 2014-05
*/
//#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION



// Python.h *must* be the first include
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "numpy/arrayobject.h"
#include <math.h>
#include <stdio.h>

#if PY_MAJOR_VERSION >= 3
  #define MOD_ERROR_VAL NULL
  #define MOD_SUCCESS_VAL(val) val
  #define MOD_INIT(name) PyMODINIT_FUNC PyInit_##name(void)
  #define MOD_DEF(ob, name, doc, methods) \
          static struct PyModuleDef moduledef = { \
            PyModuleDef_HEAD_INIT, name, doc, -1, methods, }; \
          ob = PyModule_Create(&moduledef);
#else
  #define MOD_ERROR_VAL
  #define MOD_SUCCESS_VAL(val)
  #define MOD_INIT(name) void init##name(void)
  #define MOD_DEF(ob, name, doc, methods) \
          ob = Py_InitModule3(name, methods, doc);
#endif

// Forward declarations

static PyObject * log_z(PyObject *self, PyObject *args);
static PyObject * init_kappa(PyObject *self, PyObject *args);
static PyObject * log_likelihood(PyObject *self, PyObject *args);

// Python module initialization code
static double * kappa;
static int undefined_kappa = 1;

static PyMethodDef module_functions[] = {
    {"log_z", log_z, METH_VARARGS,
     "log_z(edge_pot, node_pot, y)\n\n"
     "computes log Z, modifies node_pot and edge_pot in place."  
     },
    {"init_kappa", init_kappa, METH_VARARGS,
     "init_kappa(T_max)\n\n"
     "creates an array for kappa to be used at every subsequent log-likelihood computation."  
     },
     {"log_likelihood", log_likelihood, METH_VARARGS,
     "log_likelihood(edge_pot, node_pot, y)\n\n"
     "computes log-likelihood using the implementation for log_z."  
     },
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

MOD_INIT(chain_forwards_backwards_native)
{
    PyObject *m;
    MOD_DEF(m, "chain_forwards_backwards_native", "C implementation of the forwards-backwards algorithm",
            module_functions)
    if (m == NULL)
        return MOD_ERROR_VAL;
 	import_array();  // Must be present for NumPy.
    return MOD_SUCCESS_VAL(m);

}

// Functions available to Python

static PyObject * init_kappa(PyObject *self, PyObject *args) {
    int T_max;
    if (!PyArg_ParseTuple(args, "i",
			&T_max)) {
		return NULL;
	}
    if (T_max <= 0) {
		PyErr_SetString(PyExc_ValueError,
				"T_max must be >0");
		return NULL;
	}
    kappa = malloc(T_max * sizeof(double));
    undefined_kappa = 0;
    // what with this: 
    // delete[] kappa;
    // ???
    
	// Return None
	Py_INCREF(Py_None);
	return Py_None;
}
    
static double log_z_impl(PyArrayObject *edge_pot, PyArrayObject *node_pot, PyArrayObject *y, npy_intp T, npy_intp n_labels) {
    double log_z = 0.0;

    npy_intp t;
	for (t=0; t<T; t++) { // row of node_pot
		char *p_node_pot_t = node_pot->data + t*node_pot->strides[0]; // SB why char* ? // pointer to a row inside node_pot
        kappa[t] = 0;
        
        npy_intp l;
		for (l=0; l<n_labels; l++) { // col of node_pot
			npy_float32 *node_pot_t_l = (npy_float32*)(p_node_pot_t + l*node_pot->strides[1]);
            // if t>0 premultiply node_pot_t_l as necessary
            if (t>0) {
                char *p_node_pot_tm1 = node_pot->data + (t - 1)*node_pot->strides[0];
                char *p_edge_pot_col_l = edge_pot->data + l*edge_pot->strides[1]; // taking from edge_pot.T !
                npy_float32 a = 0.0;
                int j;
                for (j=0; j<n_labels; j++) {
                    a += (*(npy_float32*)(p_edge_pot_col_l + j*edge_pot->strides[0]))
                        * (*(npy_float32*)(p_node_pot_tm1 + j*node_pot->strides[1]));// edge_pot.T[l,j] * node_pot[t-1,j]
//                    printf("edge_pot.T[l,j]: %.2f   ", *(npy_float32*)(p_edge_pot_col_l + j*edge_pot->strides[0]));
//                    printf("node_pot[t-1,j]: %.2f   ", *(npy_float32*)(p_node_pot_tm1 + j*node_pot->strides[1]));
                }
                *node_pot_t_l *=  a;
                //printf("a_l: %.10f   ", a);

            }
			//npy_float32 node_pot_t_l = *(npy_float32*)(p_node_pot_t + l*node_pot->strides[1]);
            kappa[t] += *node_pot_t_l;
		}
		for (l=0; l<n_labels; l++) { // col of node_pot
			npy_float32 *node_pot_t_l = (npy_float32*)(p_node_pot_t + l*node_pot->strides[1]);
            *node_pot_t_l /= kappa[t];
		}
        log_z += log(kappa[t]); // this line could go into separate loop, after all of the previous has been done
        //printf("log_z: %.10f\n", log_z);

	}
    return log_z;
}

static PyObject * log_z(PyObject *self, PyObject *args) {
	PyArrayObject *edge_pot, *node_pot, *y = NULL; 
	npy_intp T, n_labels;

    if (undefined_kappa) {
        PyErr_SetString(PyExc_ValueError,
				"must initialize kappa using function init_kappa(T_max).");
		return NULL;
    }
	// Get out the arguments of *args
	if (!PyArg_ParseTuple(args, "O!O!O!",
			&PyArray_Type, &edge_pot,
			&PyArray_Type, &node_pot,
			&PyArray_Type, &y)) {
		return NULL;
	}
	if (edge_pot == NULL) return NULL;
	if (node_pot == NULL) return NULL;
	if (y == NULL) return NULL;

	// Check type of arguments
	if (
	    edge_pot->descr->type_num != NPY_FLOAT || edge_pot->nd != 2 ||
        node_pot->descr->type_num != NPY_FLOAT || node_pot->nd != 2 ||
	    y->descr->type_num != NPY_INT8 || y->nd != 1)  {
		PyErr_SetString(PyExc_ValueError,
				"node_pot, edge_pot: 2D arrays of float32; y: 1D array of int8.");
		return NULL;
	}

	// Check array dimensions
	T = node_pot->dimensions[0];
	n_labels = node_pot->dimensions[1];
	if (edge_pot->dimensions[0] != n_labels || edge_pot->dimensions[1] != n_labels ||
	    y->dimensions[0] != T) {
		PyErr_SetString(PyExc_ValueError,
				"Shape error.");
		return NULL;
	}

	return Py_BuildValue("f", log_z_impl(edge_pot, node_pot, y, T, n_labels)); // make python float32 out of C double
}

static PyObject * log_likelihood(PyObject *self, PyObject *args) {
	PyArrayObject *edge_pot, *node_pot, *y = NULL;
	npy_intp T, n_labels;

    if (undefined_kappa) {
        PyErr_SetString(PyExc_ValueError,
				"must initialize kappa using function init_kappa(T_max).");
		return NULL;
    }
	// Get out the arguments of *args
	if (!PyArg_ParseTuple(args, "O!O!O!",
			&PyArray_Type, &edge_pot,
			&PyArray_Type, &node_pot,
			&PyArray_Type, &y)) {
		return NULL;
	}
	if (node_pot == NULL) return NULL;
	if (edge_pot == NULL) return NULL;
	if (y == NULL) return NULL;

	// Check type of arguments
	if (
	    edge_pot->descr->type_num != NPY_FLOAT || edge_pot->nd != 2 ||
        node_pot->descr->type_num != NPY_FLOAT || node_pot->nd != 2 ||
	    y->descr->type_num != NPY_INT8 || y->nd != 1)  {
		PyErr_SetString(PyExc_ValueError,
				"node_pot, edge_pot: 2D arrays of float32; y: 1D array of int8.");
		return NULL;
	}

	// Check array dimensions
	T = node_pot->dimensions[0];
	n_labels = node_pot->dimensions[1];
	if (edge_pot->dimensions[0] != n_labels || edge_pot->dimensions[1] != n_labels ||
	    y->dimensions[0] != T) {
		PyErr_SetString(PyExc_ValueError,
				"Shape error.");
		return NULL;
	}
    
    double log_potential = 0.0;
    npy_intp y_t, y_tm1;
    npy_intp t;
    for (t=0; t<T; t++) { // row of node_pot
		char *p_node_pot_t = node_pot->data + t*node_pot->strides[0]; // pointer to node_pot[t,:]
        y_t = * (npy_byte*)(y->data + t*y->strides[0]);
        log_potential += log(*(npy_float32*)(p_node_pot_t + y_t*node_pot->strides[1])); // accumulate node_pot[t, y[t]]
        //printf("y(t)=%lu    ", y_t);
        //printf("log_potential: %.10f\n", log_potential);
        if (t >0) {
            y_tm1 = * (npy_byte*)(y->data + (t-1)*y->strides[0]);
            char *p_edge_pot_y_tm1_y_t = edge_pot->data + y_tm1*edge_pot->strides[0] + y_t*edge_pot->strides[1]; 
            log_potential += log(*(npy_float32*)p_edge_pot_y_tm1_y_t); // accumulate edge_pot[t-1,t]
            //printf("log_potential: %.10f\n", log_potential);
        }
    }

	return Py_BuildValue("f", log_potential - log_z_impl(edge_pot, node_pot, y, T, n_labels)); // make python float32 out of C double
}

