package com.example.corrector;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class WordCorrectorTrainingTest {

    @Test
    void maybeMisspell() {
        String misspelled = WordCorrectorTraining.maybeMisspell("option", List.of('a', 'b'));
        System.out.println(misspelled);
    }
}