use std::io;
use std::rc::Rc;
use ort::{Environment, ExecutionProvider, GraphOptimizationLevel, OrtResult, SessionBuilder, Value};
use ort::execution_providers::NNAPIExecutionProviderOptions;
use ort::session::Input;
use ndarray::prelude::*;
use ndarray::Array;
use ort::tensor::OrtOwnedTensor;

fn main()-> OrtResult<()> {
    tracing_subscriber::fmt::init();
    let mut stdout = io::stdout();
    // let mut rng = rand::thread_rng();

    let environment = Environment::builder()
        .with_name("model_test")
        // .with_execution_providers([ExecutionProvider::NNAPI(NNAPIExecutionProviderOptions{
        //     use_fp16: true,
        //     use_nchw: false,
        //     disable_cpu: true,
        //     cpu_only: false,
        // })])
        .build()?
        .into_arc();
    let session = SessionBuilder::new(&environment)?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(1)?
        .with_model_from_file("model_vision_s.onnx")?;
    let mut inputs:Vec<CowArray<f32,IxDyn>> = Vec::new();
    for Input{name,input_type,dimensions}  in session.inputs.iter(){
        println!("name: {:?}, input_type: {:?}, dimensions: {:?}",name,input_type,dimensions);
        let dimensions:Vec<usize> = dimensions.iter().map(|d| return if d.is_some() { d.unwrap() as usize } else { 1 }).collect();
        let array = CowArray::from(Array::<f32, _>::zeros(dimensions)).into_dyn();
        // let array = Rc::new(array);
        // let array = Value::from_array(session.allocator(), )?;
        inputs.push(array);
    }
    let inputs = inputs.iter().map(|input| Value::from_array(session.allocator(),input).unwrap()).collect::<Vec<_>>();
    let outputs: Vec<Value> = session.run(inputs)?;
    let output = outputs[0].try_extract()? as OrtOwnedTensor<f32, _>;
        println!("{:?}",output.view().shape());
    // println!("{:?}", inputs);



    // session.run()
    Ok(())

}
