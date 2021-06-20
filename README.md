# City Layout Generation using Artificial Neural Networks and Procedural Generation
**For more details regarding implementation**, please go [to this project's webpage](https://giodestone.github.io/projects/dissertation.html), or read [the dissertation](https://raw.githubusercontent.com/giodestone/ann-and-pg-city-layout-generator/main/Dissertation.pdf) which was ~~painfully~~ beautifully typeset using LaTeX.

The project was completed as the dissertation for my BSc (Hons) Computer Game Applications Development degree from Abertay University. The aim was to implement and evaluate how effective the generation of city layouts would be using a combination of artificial neural networks and procedural generation.

The implemented method does not generate roads. Proposed solutions include: moving one-hot encoding to an embedding layer/generator to allow more samples to be passed in, moving to a recursive model to improve accuracy, experimenting with different units.

Result images are contained in the Images folder.

The road generation is based on [Neural Turtle Graphics for Modelling City Layouts](https://github.com/nv-tlabs/NTG). Plot generation is inspired by [https://probabletrain.itch.io/city-generator](Probable Train's City Generator).

If you have any questions, message me, open an issue, or check this project's website for other ways to get in touch 🙂

## File Structure
* `encoder_decoder_rnn_roads.py` - Contains operations related to generating and encoding training sequences, network creation, and training.
* `evaluation.py` - Evaluating the network and related training configuration logic. If you're actually wanting to train a network you'd use/modify this file.
* `get_nodes_networkx.py` - Related to querying the OpenStreetMap API, processing the retrieved roads into a graph format while making sure its the correct distance.
* `plot_generation.py` - Plot detection logic, mostly implementing cyclic Dijkstra. 

## Requirements
The project was made on TensorFlow version 2.4 (though the nightly prerelease was used during development), to speed up training you need to configure your PC to use GPU acceleration.