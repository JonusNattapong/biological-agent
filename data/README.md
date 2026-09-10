# MaleCNS data

NeuroFly can ingest the official **MaleCNS v1.0** bulk connectome published by HHMI Janelia.
The dataset is third-party data and is **not vendored in this repository**.

Core files used by NeuroFly:

- `body-annotations-male-cns-v1.0-minconf-0.5.feather` — curated neuron annotations (~13 MB)
- `body-neurotransmitters-male-cns-v1.0.feather` — aggregate neurotransmitter predictions (~42 MB)
- `connectome-weights-male-cns-v1.0-minconf-0.5.feather` — full body-to-body connection graph (~1.1 GB)

Official project/download page: `https://male-cns.janelia.org/download/`

The official bulk objects live under:

`gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/`

Use the helper script from the repository root:

```powershell
python scripts/download_malecns.py
python scripts/download_malecns.py --weights
```

The first command downloads annotations and neurotransmitter metadata. The second also downloads the large full connection graph.
Files are stored in `data/raw/` and ignored by Git.

## Scientific provenance

`male-cns:v1.0` is the biological connectivity source. NeuroFly's LIF neuron equations,
weight transforms, sensory current injection, coarse functional-region mapping, and motor
decoding are simulation assumptions layered on top of that reconstructed graph.
They are not direct electrophysiological measurements.

MaleCNS is distributed by its authors under CC-BY; retain dataset attribution when publishing derived results.
