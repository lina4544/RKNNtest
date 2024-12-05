import numpy as np
import torch
from rknn.api import RKNN
from transformer import TransformerModel  # Import Transformer model
from load_datasets import src_vocab, tgt_vocab  # Import vocabularies
import os


def export_transformer_model():
    """
    Function to export the Transformer model as a TorchScript file.
    """
    input_dim = len(src_vocab)  # Vocabulary size from source
    embed_dim = 512
    num_heads = 8
    ff_dim = 2048
    num_layers = 6
    dropout = 0.1

    # Initialize Transformer model
    net = TransformerModel(input_dim, embed_dim, num_heads, ff_dim, num_layers, dropout)
    net.eval()

    # Export model to TorchScript
    dummy_input = torch.randint(0, input_dim, (1, 50))  # Example input: batch_size=1, seq_len=50
    trace_model = torch.jit.trace(net, dummy_input)
    trace_model.save('./transformer.pt')


def show_outputs(output):
    """
    Decodes the Transformer model output and prints the translated sequence.
    """
    decoded_sequence = output.argmax(dim=-1).squeeze(0).tolist()
    translated_sentence = " ".join(
        tgt_vocab.itos[idx] for idx in decoded_sequence if idx not in {tgt_vocab['<pad>'], tgt_vocab['<eos>']}
    )
    print(f"Translated Sentence: {translated_sentence}")


if __name__ == '__main__':
    # Check if the TorchScript model exists, export if not
    model_path = './transformer.pt'
    if not os.path.exists(model_path):
        export_transformer_model()

    # Transformer input shape: [batch_size, seq_len]
    input_size_list = [[1, 50]]
    rknn = RKNN(verbose=True)

    # Configure RKNN
    print('--> Configuring RKNN model')
    rknn.config(target_platform='rk3588')
    print('done')

    # Load TorchScript model
    print('--> Loading Transformer model')
    ret = rknn.load_pytorch(model=model_path, input_size_list=input_size_list)
    if ret != 0:
        print('Failed to load model!')
        exit(ret)
    print('done')

    # Build RKNN model
    print('--> Building RKNN model')
    ret = rknn.build(do_quantization=False)  # Quantization optional for NLP tasks
    if ret != 0:
        print('Failed to build model!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Exporting RKNN model')
    ret = rknn.export_rknn('./transformer.rknn')
    if ret != 0:
        print('Failed to export RKNN model!')
        exit(ret)
    print('done')

    # Prepare input data
    print('--> Preparing input data')
    src_sentence = "Hallo Welt!"  # Example German sentence
    src_indices = [src_vocab[token] for token in src_sentence.split()]
    src_tensor = torch.tensor([src_vocab['<bos>']] + src_indices + [src_vocab['<eos>']], dtype=torch.long).unsqueeze(0)
    print('done')

    # Initialize RKNN runtime environment
    print('--> Initializing RKNN runtime environment')
    ret = rknn.init_runtime(target)
    if ret != 0:
        print('Failed to initialize runtime!')
        exit(ret)
    print('done')

    # Run inference
    print('--> Running inference on Transformer model')
    outputs = rknn.inference(inputs=[src_tensor.numpy()])
    np.save('./transformer_output.npy', outputs[0])  # Save outputs for analysis
    show_outputs(torch.tensor(outputs[0]))  # Decode and display output
    print('done')

    # Release RKNN resources
    rknn.release()
