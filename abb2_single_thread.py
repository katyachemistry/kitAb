#!/usr/bin/env python3

import os
import argparse
import torch
torch.cuda.is_available = lambda: False
from ImmuneBuilder import ABodyBuilder2

# Make CUDA invisible and limit PyTorch threads
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
torch.set_num_threads(1)

# Argument parsing
parser = argparse.ArgumentParser(description="Predict antibody structure from heavy and light chain sequences.")
parser.add_argument("name", help="Name of the antibody/output file prefix")
parser.add_argument("heavy", help="Heavy chain amino acid sequence")
parser.add_argument("light", help="Light chain amino acid sequence")
parser.add_argument("--output-dir", default="structures", help="Output directory for PDB files (default: structures)")

args = parser.parse_args()

# Initialize predictor
predictor = ABodyBuilder2()
print("The device is", predictor.device)

# Prepare sequences dictionary
sequences = {
    'H': args.heavy,
    'L': args.light
}
os.makedirs(args.output_dir, exist_ok=True)
output_file = f"{args.output_dir}/{args.name.replace('|', '_')}.pdb"

try:
    antibody = predictor.predict(sequences)
except Exception:
    print(f"Have not been able to PREDICT {output_file}")
    
try:
    antibody.save(output_file, n_threads=1)
except Exception:
    print(f"Have not been able to SAVE {output_file}")

