class L1Layer extends tf.layers.Layer {
    constructor() {
        super({});
    }
    // In this case, the output is a scalar.
    computeOutputShape(inputShape) { return []; }

    // call() is where we do the computation.
    call(inputs, kwargs) { 
        console.log("Inputs", inputs)
        const left = inputs['left'].shape
        const right = inputs['right'].shape
        return tf.abs(tf.sub(left, right));
    }

    // Every layer needs a unique name.
    static get className() { 
        return 'L1'; 
    }
}

tf.serialization.registerClass(L1Layer);


export function L1() {
    return new L1Layer();
}