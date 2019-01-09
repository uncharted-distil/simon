# vNEXT
* Heterogeneous acceleration (multiple CPUs, GPUs)
* ROC optimization of p_threshold on a per-class basis
* more complete set of metrics in evaluation function (precision, recall, F1...)
* .csv data processing utility

# v1.2.2
* activation in last fully-connected layer now an optional keyword argument
to .generate_model/.generate_transfer_model
# v1.2.1
* GPU acceleration
* minor improvements
# v1.2.0
* Added transfer learning capability via Classifier.generate_transfer_model method
* Includes a new method, that guesses categorical/ordinal type classification based on some fixed statistical/proportion rules
* Includes pretrained models of base types + categorical/ordinal and base types + categorical/ordinal + 7 geographical categories, i.e. 'address', 'boolean', 'categorical', 'city', 'country', 'country_code', 'datetime', 'email', 'float', 'int', 'latitude', 'longitude', 'ordinal', 'phone', 'postal_code', 'state', 'text', 'uri'
* Added a rest/docker interface for deploying classifier

# v1.1.0
* First pip-installable version
* CNN+LSTM learning for text classification
* Includes pretrained model Base.pkl for semantic classification of 9 base types - 'address', 'boolean', 'datetime', 'email', 'float', 'int', 'phone', 'text', 'uri'
