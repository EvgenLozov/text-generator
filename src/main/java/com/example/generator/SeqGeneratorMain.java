package com.example.generator;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class SeqGeneratorMain {
    public static void main(String[] args) throws IOException {
        String modelFileName = "model.bin";
        ComputationGraph graph = ModelSerializer.restoreComputationGraph(new File(modelFileName));
        Map<Character,Integer> char2index = new HashMap<>();

        Function<Character, List<Integer>> f = new FunctionBuilder<>(Function.<Integer>identity())
                .andThen(charNumber -> Nd4j.create(new double[]{charNumber}).reshape(1, 1))
                .andThen(graph::rnnTimeStep)
                .andThen(indArrays -> indArrays[0].toFloatVector())
                .andThen(floats -> IntStream.range(0, 148)
                        .boxed()
                        .collect(Collectors.toMap(Function.identity(), i -> floats[i]))
                        .entrySet()
                        .stream()
                        .sorted((e1, e2) -> Float.compare(e2.getValue(), e1.getValue()))
                        .limit(1)
                        .findFirst()
                        .get()
                        .getKey()
                )
                .<Integer, List<Integer>>wrap(generator -> integer -> {
                    int charIndex = 1;
                    List<Integer> indexes = new ArrayList<>();
                    indexes.add(charIndex);

                    for (int i = 0; i < 100; i++) {
                        charIndex = generator.apply(charIndex);
                        indexes.add(charIndex);
                    }

                    return indexes;
                })
                .andThen(integers -> integers.stream()
                        .map(char2index::get)
                        .collect(Collectors.toList())
                )
                .<Character>compose(char2index::get)
                .getF();


    }

    private static Map<Integer, Integer> uniqueCharsIndices(File file) throws IOException {
        AtomicInteger index = new AtomicInteger(2);

        return Files.lines(file.toPath())
                .filter(line -> line.chars().noneMatch(i -> i > 256))
                .flatMapToInt(String::chars)
                .distinct()
                .boxed()
                .collect(Collectors.toMap(Function.identity(), character -> index.getAndIncrement()));
    }
}
