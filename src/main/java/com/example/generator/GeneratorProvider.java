package com.example.generator;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class GeneratorProvider {

    public static Function<Integer, List<Integer>> generator(Map<Integer, Integer> char2index, ComputationGraph graph, int endIndex){


        return new FunctionBuilder<>(Function.<Integer>identity())
                .andThen(charNumber -> Nd4j.create(new double[]{charNumber}).reshape(1, 1))
                .andThen(graph::rnnTimeStep)
                .andThen(indArrays -> indArrays[0].toFloatVector())
                .andThen(floats -> {
                    RandomCollection<Integer> rc = new RandomCollection<>();
                    IntStream.range(0, char2index.size() + 2)
                            .boxed()
                            .forEach(integer -> rc.add(floats[integer], integer));

                    return rc.next();
                })
                .<Integer, List<Integer>>wrap(generator -> integer -> {
                    int charIndex = 0;
                    List<Integer> indexes = new ArrayList<>();
                    indexes.add(charIndex);

                    while (charIndex != endIndex && indexes.size() < 50){
                        charIndex = generator.apply(integer);
                        indexes.add(charIndex);
                    }

                    return indexes;
                })
                .getF();
    }
}
