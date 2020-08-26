package com.example.corrector;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;

import java.io.File;
import java.io.IOException;

public class WordsCorrectorModel {
    public static ComputationGraph load(File modelFile) throws IOException {
        return ModelSerializer.restoreComputationGraph(modelFile);
    }

    public static ComputationGraph build(int uniqueCharsCount) {
        //TODO
        return null;
    }
}
