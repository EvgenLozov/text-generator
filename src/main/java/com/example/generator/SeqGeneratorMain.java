package com.example.generator;

import org.deeplearning4j.nn.graph.ComputationGraph;
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
        String examplesFileName = "eng_wikipedia_2010_300K-sentences.txt";
        String modelFileName = "model.bin";

        ComputationGraph graph = Models.load(new File(modelFileName));
        File file = new File(examplesFileName);

        Map<Integer, Integer> char2index = uniqueCharsIndices(file);
        Map<Integer, Integer> index2char = new HashMap<>();
        for (Map.Entry<Integer, Integer> entry : char2index.entrySet()) {
            index2char.put(entry.getValue(), entry.getKey());
        }

        Function<Integer, List<Integer>> f = new FunctionBuilder<>(Function.<Integer>identity())
                .andThen(charNumber -> Nd4j.create(new double[]{charNumber}).reshape(1, 1))
                .andThen(graph::rnnTimeStep)
                .andThen(indArrays -> indArrays[0].toFloatVector())
                .andThen(floats -> IntStream.range(0, char2index.size() + 2)
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
                })
                .andThen(integers -> integers.stream()
                        .filter(i -> i > 1)
                        .map(index2char::get)
                        .collect(Collectors.toList()))
                .getF();

        for (int i = 0; i < 10; i++) {
            f.apply(0).forEach(c -> System.out.print((char) c.intValue()));
            System.out.println();
            graph.rnnClearPreviousState();
        }

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
