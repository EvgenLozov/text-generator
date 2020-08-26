package com.example.generator;

import com.example.FunctionBuilder;
import com.example.Pair;
import com.example.RandomCollection;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class GeneratorProvider  {

    Function<Integer, List<Integer>> get(ComputationGraph graph, int uniqueCharsCount){
        return new FunctionBuilder<>(Function.<Integer>identity())
                .andThen(charNumber -> Nd4j.create(new double[]{charNumber}).reshape(1, 1))
                .andThen(graph::rnnTimeStep)
                .andThen(indArrays -> indArrays[0].toFloatVector())
                .andThen(floats -> IntStream.range(0, uniqueCharsCount)
                        .boxed()
                        .collect(Collectors.toMap(Function.identity(), i -> floats[i]))
                        .entrySet()
                        .stream()
                        .sorted((e1, e2) -> Float.compare(e2.getValue(), e1.getValue()))
                        .limit(5)
                        .map(e -> new Pair<>(e.getKey(), e.getValue()))
                        .collect(Collectors.toList())
                )
                .andThen(ind2Weight -> {
                    RandomCollection<Integer> rc = new RandomCollection<>();
                    ind2Weight.forEach(e -> rc.add(e.getSecond(), e.getFirst()));

                    return rc.next();
                })
                .<Integer, List<Integer>>wrap(generator -> integer -> {
                    int charIndex = 0;
                    List<Integer> indexes = new ArrayList<>();
                    indexes.add(charIndex);

                    while (charIndex != 1){
                        charIndex = generator.apply(charIndex);
                        indexes.add(charIndex);
                    }

                    return indexes;
                }).getF();
    }
}
