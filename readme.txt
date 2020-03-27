strip_unused.py 用來縮減已凍結的model(.pb檔)  
python strip_unused.py --input_graph=frozen_model.pb --output_graph=new.pb --input_node_names="Placeholder" --output_node_names="cnn_1/Softmax" --input_binary=true
此指令也可用來確定輸入與輸出點是否存在model中

frozen.py 用來凍結model
參數為
--model_dir MODEL_DIR
                        Model folder to export
--output_node_names OUTPUT_NODE_NAMES
                        The name of the output nodes, comma separated.

python frozen.py --model_dir="300/" --output_node_names="cnn_1/dense_3/BiasAdd"

python strip_unused.py --input_graph=cnn_frozen_model.pb --output_graph=new.pb --input_node_names="Placeholder" --output_node_names="cnn_1/dense_3" --input_binary=true