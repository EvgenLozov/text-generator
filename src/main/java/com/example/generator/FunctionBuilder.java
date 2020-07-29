package com.example.generator;

import java.util.function.BiFunction;
import java.util.function.Function;

public class FunctionBuilder<A,B> {

    private Function<A,B> f;

    public FunctionBuilder(Function<A, B> f) {
        this.f = f;
    }

    public <C> FunctionBuilder<A,C> andThen(Function<B,C> fun){

        return new FunctionBuilder<>(f.andThen(fun));
    }

    public <A1> FunctionBuilder<A1,B> compose(Function<A1,A> composeFun){
        return new FunctionBuilder<>(f.compose(composeFun));
    }

    public <C> FunctionBuilder<A,C> andThenArg(BiFunction<A,B,C> fun){
        return new FunctionBuilder<>( a -> {
            B b = f.apply(a);

            return fun.apply(a,b);
        } );
    }

    public <A1,B1> FunctionBuilder<A1,B1> wrap(Function<Function<A,B>,Function<A1,B1>> wrapperFunction){
        return new FunctionBuilder<>(wrapperFunction.apply(f));
    }

    public  FunctionBuilder<A,B> decorate(Function<Function<A,B>,Function<A,B>> wrapperFunction){
        return new FunctionBuilder<>(wrapperFunction.apply(f));
    }

    public Function<A, B> getF() {
        return f;
    }
}
