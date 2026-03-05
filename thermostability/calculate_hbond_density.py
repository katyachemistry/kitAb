#!/usr/bin/env python3
"""
Command-line interface for calculating global hydrogen bond density.

Usage:
    python calculate_hbond_density.py <pdb_file> <sasa_file> [options]
"""

import argparse
import sys
from pathlib import Path

# Add FASTAb directory to path to import descriptors
# This allows running from the FASTAb directory or from anywhere
fastab_dir = Path(__file__).parent.parent
sys.path.insert(0, str(fastab_dir))

from thermostability.descriptors import (
    calculate_global_hbond_density,
    calculate_global_hbond_density_average
)


def main():
    parser = argparse.ArgumentParser(
        description='Calculate global hydrogen bond density from antibody structures and SASA files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calculate average hydrogen bond density
  python calculate_hbond_density.py structure.pdb structure.sasa --average

  # Calculate per-residue densities and save to file
  python calculate_hbond_density.py structure.pdb structure.sasa --output results.txt

  # Calculate per-residue densities and print to stdout
  python calculate_hbond_density.py structure.pdb structure.sasa
        """
    )
    
    parser.add_argument(
        'pdb_file',
        type=str,
        help='Path to PDB structure file'
    )
    
    parser.add_argument(
        'sasa_file',
        type=str,
        help='Path to SASA file (FreeSASA format)'
    )
    
    parser.add_argument(
        '--average',
        action='store_true',
        help='Calculate and output only the average hydrogen bond density'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output file path (default: print to stdout)'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['table', 'csv', 'json'],
        default='table',
        help='Output format (default: table)'
    )
    
    args = parser.parse_args()
    
    # Modify SASA file path to use "_full" suffix
    if args.sasa_file:
        sasa_path = Path(args.sasa_file)
        # Check if "_full" is already in the filename
        if "_full" not in sasa_path.stem:
            # Insert "_full" before the extension
            new_stem = sasa_path.stem + "_full"
            args.sasa_file = str(sasa_path.with_name(new_stem + sasa_path.suffix))
    
    # Check if files exist
    if not Path(args.pdb_file).exists():
        print(f"Error: PDB file not found: {args.pdb_file}", file=sys.stderr)
        sys.exit(1)
    
    if not Path(args.sasa_file).exists():
        print(f"Error: SASA file not found: {args.sasa_file}", file=sys.stderr)
        sys.exit(1)
    
    # Calculate hydrogen bond density
    try:
        if args.average:
            # Calculate average only
            avg_density = calculate_global_hbond_density_average(args.pdb_file, args.sasa_file)
            
            output_text = f"Average global hydrogen bond density: {avg_density:.4f}\n"
            
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(output_text)
                print(f"Results written to {args.output}")
            else:
                print(output_text, end='')
        
        else:
            # Calculate per-residue densities
            residue_densities, _, _ = calculate_global_hbond_density(args.pdb_file, args.sasa_file)
            
            if args.format == 'table':
                output_lines = [
                    f"Hydrogen bond density per residue (weighted by inverse SASA)",
                    f"Total residues with H-bonds: {len(residue_densities)}",
                    f"Total weighted H-bonds: {sum(residue_densities.values()):.4f}",
                    f"Average per residue: {sum(residue_densities.values())/len(residue_densities) if residue_densities else 0:.4f}",
                    "",
                    "Residue\tChain\tNumber\tDensity"
                ]
                
                # Sort by residue number and chain for better readability
                sorted_residues = sorted(
                    residue_densities.items(),
                    key=lambda x: (x[0][2], x[0][1])  # Sort by chain, then residue number
                )
                
                for (res_name, res_num, chain), density in sorted_residues:
                    output_lines.append(f"{res_name}\t{chain}\t{res_num}\t{density:.4f}")
            
            elif args.format == 'csv':
                output_lines = [
                    "residue_name,chain,residue_number,density"
                ]
                
                sorted_residues = sorted(
                    residue_densities.items(),
                    key=lambda x: (x[0][2], x[0][1])
                )
                
                for (res_name, res_num, chain), density in sorted_residues:
                    output_lines.append(f"{res_name},{chain},{res_num},{density:.4f}")
            
            elif args.format == 'json':
                import json
                output_dict = {
                    'total_residues': len(residue_densities),
                    'total_weighted_hbonds': sum(residue_densities.values()),
                    'average_density': sum(residue_densities.values())/len(residue_densities) if residue_densities else 0.0,
                    'residues': [
                        {
                            'residue_name': res_name,
                            'chain': chain,
                            'residue_number': res_num,
                            'density': density
                        }
                        for (res_name, res_num, chain), density in sorted(
                            residue_densities.items(),
                            key=lambda x: (x[0][2], x[0][1])
                        )
                    ]
                }
                output_lines = [json.dumps(output_dict, indent=2)]
            
            output_text = '\n'.join(output_lines) + '\n'
            
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(output_text)
                print(f"Results written to {args.output}")
            else:
                print(output_text, end='')
    
    except Exception as e:
        print(f"Error calculating hydrogen bond density: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

