"""Interfaces common to all Neural Modules and Models."""
import hashlib
import inspect
import traceback
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import total_ordering
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import hydra
import wrapt
from huggingface_hub import HfApi, HfFolder, ModelFilter, hf_hub_download
from huggingface_hub.hf_api import ModelInfo
from omegaconf import DictConfig, OmegaConf


class Typing(ABC):
    """
    An interface which endows module with neural types
    """

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        """Define these to enable input neural type checks"""
        return None

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Define these to enable output neural type checks"""
        return None

    def _validate_input_types(self, input_types=None, ignore_collections=False, **kwargs):
        """
        This function does a few things.

        1) It ensures that len(self.input_types <non-optional>) <= len(kwargs) <= len(self.input_types).
        2) For each (keyword name, keyword value) passed as input to the wrapped function:
            - Check if the keyword name exists in the list of valid self.input_types names.
            - Check if keyword value has the `neural_type` property.
                - If it does, then perform a comparative check and assert that neural types
                    are compatible (SAME or GREATER).
            - Check if keyword value is a container type (list or tuple). If yes,
                then perform the elementwise test of neural type above on each element
                of the nested structure, recursively.

        Args:
            input_types: Either the `input_types` defined at class level, or the local function
                overridden type definition.
            ignore_collections: For backward compatibility, container support can be disabled explicitly
                using this flag. When set to True, all nesting is ignored and nest-depth checks are skipped.
            kwargs: Dictionary of argument_name:argument_value pairs passed to the wrapped
                function upon call.
        """
        # TODO: Properly implement this
        if input_types is not None:
            # Precompute metadata
            metadata = TypecheckMetadata(original_types=input_types, ignore_collections=ignore_collections)

            total_input_types = len(input_types)
            mandatory_input_types = len(metadata.mandatory_types)

            # Allow number of input arguments to be <= total input neural types.
            if len(kwargs) < mandatory_input_types or len(kwargs) > total_input_types:
                raise TypeError(
                    f"Number of input arguments provided ({len(kwargs)}) is not as expected. Function has "
                    f"{total_input_types} total inputs with {mandatory_input_types} mandatory inputs."
                )

            for key, value in kwargs.items():
                # Check if keys exists in the defined input types
                if key not in input_types:
                    raise TypeError(
                        f"Input argument {key} has no corresponding input_type match. "
                        f"Existing input_types = {input_types.keys()}"
                    )

                # Perform neural type check
                if hasattr(value, 'neural_type') and not metadata.base_types[key].compare(value.neural_type) in (
                    NeuralTypeComparisonResult.SAME,
                    NeuralTypeComparisonResult.GREATER,
                ):
                    error_msg = [
                        f"{input_types[key].compare(value.neural_type)} :",
                        f"Input type expected : {input_types[key]}",
                        f"Input type found : {value.neural_type}",
                        f"Argument: {key}",
                    ]
                    for i, dict_tuple in enumerate(metadata.base_types[key].elements_type.type_parameters.items()):
                        error_msg.insert(i + 2, f'  input param_{i} : {dict_tuple[0]}: {dict_tuple[1]}')
                    for i, dict_tuple in enumerate(value.neural_type.elements_type.type_parameters.items()):
                        error_msg.append(f'  input param_{i} : {dict_tuple[0]}: {dict_tuple[1]}')
                    raise TypeError("\n".join(error_msg))

                # Perform input ndim check
                if hasattr(value, 'shape'):
                    value_shape = value.shape
                    type_shape = metadata.base_types[key].axes
                    name = key

                    if type_shape is not None and len(value_shape) != len(type_shape):
                        raise TypeError(
                            f"Input shape mismatch occured for {name} in module {self.__class__.__name__} : \n"
                            f"Input shape expected = {metadata.base_types[key].axes} | \n"
                            f"Input shape found : {value_shape}"
                        )

                # Perform recursive neural type check for homogeneous elements
                elif isinstance(value, list) or isinstance(value, tuple):
                    for ind, val in enumerate(value):
                        """
                        This initiates a DFS, tracking the depth count as it goes along the nested structure.
                        Initial depth is 1 as we consider the current loop to be the 1st step inside the nest.
                        """
                        self.__check_neural_type(val, metadata, depth=1, name=key)

    def _attach_and_validate_output_types(self, out_objects, ignore_collections=False, output_types=None):
        """
        This function does a few things.

        1) It ensures that len(out_object) == len(self.output_types).
        2) If the output is a tensor (or list/tuple of list/tuple ... of tensors), it
            attaches a neural_type to it. For objects without the neural_type attribute,
            such as python objects (dictionaries and lists, primitive data types, structs),
            no neural_type is attached.

        Note: tensor.neural_type is only checked during _validate_input_types which is
        called prior to forward().

        Args:
            output_types: Either the `output_types` defined at class level, or the local function
                overridden type definition.
            ignore_collections: For backward compatibility, container support can be disabled explicitly
                using this flag. When set to True, all nesting is ignored and nest-depth checks are skipped.
            out_objects: The outputs of the wrapped function.
        """
        # TODO: Properly implement this
        if output_types is not None:
            # Precompute metadata
            metadata = TypecheckMetadata(original_types=output_types, ignore_collections=ignore_collections)
            out_types_list = list(metadata.base_types.items())
            mandatory_out_types_list = list(metadata.mandatory_types.items())

            # First convert all outputs to list/tuple format to check correct number of outputs
            if isinstance(out_objects, (list, tuple)):
                out_container = out_objects  # can be any rank nested structure
            else:
                out_container = [out_objects]

            # If this neural type has a *single output*, with *support for nested outputs*,
            # then *do not* perform any check on the number of output items against the number
            # of neural types (in this case, 1).
            # This is done as python will *not* wrap a single returned list into a tuple of length 1,
            # instead opting to keep the list intact. Therefore len(out_container) in such a case
            # is the length of all the elements of that list - each of which has the same corresponding
            # neural type (defined as the singular container type).
            if metadata.is_singular_container_type:
                pass

            # In all other cases, python will wrap multiple outputs into an outer tuple.
            # Allow number of output arguments to be <= total output neural types and >= mandatory outputs.

            elif len(out_container) > len(out_types_list) or len(out_container) < len(mandatory_out_types_list):
                raise TypeError(
                    "Number of output arguments provided ({}) is not as expected. It should be larger than {} and less than {}.\n"
                    "This can be either because insufficient/extra number of output NeuralTypes were provided,"
                    "or the provided NeuralTypes {} should enable container support "
                    "(add '[]' to the NeuralType definition)".format(
                        len(out_container), len(out_types_list), len(mandatory_out_types_list), output_types
                    )
                )

            # Attach types recursively, if possible
            if not isinstance(out_objects, tuple) and not isinstance(out_objects, list):
                # Here, out_objects is a single object which can potentially be attached with a NeuralType
                try:
                    out_objects.neural_type = out_types_list[0][1]
                except Exception:
                    pass

                # Perform output ndim check
                if hasattr(out_objects, 'shape'):
                    value_shape = out_objects.shape
                    type_shape = out_types_list[0][1].axes
                    name = out_types_list[0][0]

                    if type_shape is not None and len(value_shape) != len(type_shape):
                        raise TypeError(
                            f"Output shape mismatch occured for {name} in module {self.__class__.__name__} : \n"
                            f"Output shape expected = {type_shape} | \n"
                            f"Output shape found : {value_shape}"
                        )

            elif metadata.is_singular_container_type:
                # If only a single neural type is provided, and it defines a container nest,
                # then all elements of the returned list/tuple are assumed to belong to that
                # singular neural type.
                # As such, the "current" depth inside the DFS loop is counted as 1,
                # and subsequent nesting will increase this count.

                # NOTE:
                # As the flag `is_singular_container_type` will activate only for
                # the case where there is 1 output type defined with container nesting,
                # this is a safe assumption to make.
                depth = 1

                # NOTE:
                # A user may chose to explicitly wrap the single output list within an explicit tuple
                # In such a case we reduce the "current" depth to 0 - to acknowledge the fact that
                # the actual nest exists within a wrapper tuple.
                if len(out_objects) == 1 and type(out_objects) == tuple:
                    depth = 0

                for ind, res in enumerate(out_objects):
                    self.__attach_neural_type(res, metadata, depth=depth, name=out_types_list[0][0])
            else:
                # If more then one item is returned in a return statement, python will wrap
                # the output with an outer tuple. Therefore there must be a 1:1 correspondence
                # of the output_neural type (with or without nested structure) to the actual output
                # (whether it is a single object or a nested structure of objects).
                # Therefore in such a case, we "start" the DFS at depth 0 - since the recursion is
                # being applied on 1 neural type : 1 output struct (single or nested output).
                # Since we are guarenteed that the outer tuple will be built by python,
                # assuming initial depth of 0 is appropriate.
                for ind, res in enumerate(out_objects):
                    self.__attach_neural_type(res, metadata, depth=0, name=out_types_list[ind][0])

    def __check_neural_type(self, obj, metadata: TypecheckMetadata, depth: int, name: str = None):
        """
        Recursively tests whether the obj satisfies the semantic neural type assertion.
        Can include shape checks if shape information is provided.

        Args:
            obj: Any python object that can be assigned a value.
            metadata: TypecheckMetadata object.
            depth: Current depth of recursion.
            name: Optional name used of the source obj, used when an error occurs.
        """
        if isinstance(obj, tuple) or isinstance(obj, list):
            for elem in obj:
                self.__check_neural_type(elem, metadata, depth + 1, name=name)
            return  # after processing nest, return to avoid testing nest itself

        type_val = metadata.base_types[name]

        # If nest depth doesnt match neural type structure depth, raise an error
        if not metadata.ignore_collections and depth != metadata.container_depth[name]:
            raise TypeError(
                "While checking input neural types,\n"
                "Nested depth of value did not match container specification:\n"
                f"Current nested depth of NeuralType '{name}' ({type_val}): {depth}\n"
                f"Expected nested depth : {metadata.container_depth[name]}"
            )

        if hasattr(obj, 'neural_type') and not type_val.compare(obj.neural_type) in (
            NeuralTypeComparisonResult.SAME,
            NeuralTypeComparisonResult.GREATER,
        ):
            raise TypeError(
                f"{type_val.compare(obj.neural_type)} : \n"
                f"Input type expected = {type_val} | \n"
                f"Input type found : {obj.neural_type}"
            )

        # Perform input ndim check
        if hasattr(obj, 'shape'):
            value_shape = obj.shape
            type_shape = type_val.axes

            if type_shape is not None and len(value_shape) != len(type_shape):
                raise TypeError(
                    f"Input shape mismatch occured for {name} in module {self.__class__.__name__} : \n"
                    f"Input shape expected = {type_shape} | \n"
                    f"Input shape found : {value_shape}"
                )

    def __attach_neural_type(self, obj, metadata: TypecheckMetadata, depth: int, name: str = None):
        """
        Recursively attach neural types to a given object - as long as it can be assigned some value.

        Args:
            obj: Any python object that can be assigned a value.
            metadata: TypecheckMetadata object.
            depth: Current depth of recursion.
            name: Optional name used of the source obj, used when an error occurs.
        """
        if isinstance(obj, tuple) or isinstance(obj, list):
            for elem in obj:
                self.__attach_neural_type(elem, metadata, depth=depth + 1, name=name)
            return  # after processing nest, return to avoid argument insertion into nest itself

        type_val = metadata.base_types[name]

        # If nest depth doesnt match neural type structure depth, raise an error
        if not metadata.ignore_collections and depth != metadata.container_depth[name]:
            raise TypeError(
                "While attaching output neural types,\n"
                "Nested depth of value did not match container specification:\n"
                f"Current nested depth of NeuralType '{name}' ({type_val}): {depth}\n"
                f"Expected nested depth : {metadata.container_depth[name]}"
            )

        try:
            obj.neural_type = type_val
        except Exception:
            pass

        # Perform output ndim check
        if hasattr(obj, 'shape'):
            value_shape = obj.shape
            type_shape = type_val.axes

            if type_shape is not None and len(value_shape) != len(type_shape):
                raise TypeError(
                    f"Output shape mismatch occured for {name} in module {self.__class__.__name__} : \n"
                    f"Output shape expected = {type_shape} | \n"
                    f"Output shape found : {value_shape}"
                )


class Serialization(ABC):
    @classmethod
    def from_config_dict(cls, config: 'DictConfig', trainer: Optional['Trainer'] = None):
        """Instantiates object using DictConfig-based configuration"""
        # Resolve the config dict
        if _HAS_HYDRA:
            if isinstance(config, DictConfig):
                config = OmegaConf.to_container(config, resolve=True)
                config = OmegaConf.create(config)
                OmegaConf.set_struct(config, True)

            config = maybe_update_config_version(config)

        # Hydra 0.x API
        if ('cls' in config or 'target' in config) and 'params' in config and _HAS_HYDRA:
            # regular hydra-based instantiation
            instance = hydra.utils.instantiate(config=config)
        # Hydra 1.x API
        elif '_target_' in config and _HAS_HYDRA:
            # regular hydra-based instantiation
            instance = hydra.utils.instantiate(config=config)
        else:
            instance = None
            prev_error = ""
            # Attempt class path resolution from config `target` class (if it exists)
            if 'target' in config:
                target_cls = config["target"]  # No guarantee that this is a omegaconf class
                imported_cls = None
                try:
                    # try to import the target class
                    imported_cls = import_class_by_path(target_cls)
                    # if calling class (cls) is subclass of imported class,
                    # use subclass instead
                    if issubclass(cls, imported_cls):
                        imported_cls = cls
                    accepts_trainer = Serialization._inspect_signature_for_trainer(imported_cls)
                    if accepts_trainer:
                        instance = imported_cls(cfg=config, trainer=trainer)
                    else:
                        instance = imported_cls(cfg=config)
                except Exception as e:
                    # record previous error
                    tb = traceback.format_exc()
                    prev_error = f"Model instantiation failed!\nTarget class:\t{target_cls}" f"\nError(s):\t{e}\n{tb}"
                    logging.debug(prev_error + "\nFalling back to `cls`.")

            # target class resolution was unsuccessful, fall back to current `cls`
            if instance is None:
                try:
                    accepts_trainer = Serialization._inspect_signature_for_trainer(cls)
                    if accepts_trainer:
                        instance = cls(cfg=config, trainer=trainer)
                    else:
                        instance = cls(cfg=config)

                except Exception as e:
                    # report saved errors, if any, and raise
                    if prev_error:
                        logging.error(prev_error)
                    raise e

        if not hasattr(instance, '_cfg'):
            instance._cfg = config
        return instance

    def to_config_dict(self) -> 'DictConfig':
        """Returns object's configuration to config dictionary"""
        if hasattr(self, '_cfg') and self._cfg is not None:
            # Resolve the config dict
            if _HAS_HYDRA and isinstance(self._cfg, DictConfig):
                config = OmegaConf.to_container(self._cfg, resolve=True)
                config = OmegaConf.create(config)
                OmegaConf.set_struct(config, True)

                config = maybe_update_config_version(config)

            self._cfg = config

            return self._cfg
        else:
            raise NotImplementedError(
                'to_config_dict() can currently only return object._cfg but current object does not have it.'
            )

    @classmethod
    def _inspect_signature_for_trainer(cls, check_cls):
        if hasattr(check_cls, '__init__'):
            signature = inspect.signature(check_cls.__init__)
            if 'trainer' in signature.parameters:
                return True
            else:
                return False
        else:
            return False