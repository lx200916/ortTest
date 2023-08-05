use std::{io, default, array};
use std::rc::Rc;
use ort::sys::OrtAllocator;
use ort::{Environment, ExecutionProvider, GraphOptimizationLevel, OrtResult, SessionBuilder, Value, Session};
use ort::execution_providers::NNAPIExecutionProviderOptions;
use ort::session::Input;
use ndarray::prelude::*;
use ndarray::Array;
use ort::tensor::{IntoTensorElementDataType, OrtOwnedTensor};
use std::env;
enum InputTensor {
    Int64(Array<i64,IxDyn>),
    Float32(Array<f32,IxDyn>),
}
impl<'a> InputTensor {
    fn to_tensor(&'a self,allocator:*mut OrtAllocator)->Value<'a> {

          match self {
            InputTensor::Int64(array) => {
                let array = Box::leak(Box::new(CowArray::from(array)));
              Value::from_array(allocator,array).unwrap()
            },
            InputTensor::Float32(array) => {
                let array = Box::leak(Box::new(CowArray::from(array)));
                Value::from_array(allocator,array).unwrap()
            },
        }

    }
    
}

impl Clone for InputTensor {
    fn clone(&self) -> Self {
        match self {
            InputTensor::Int64(array) => InputTensor::Int64(array.clone()),
            InputTensor::Float32(array) => InputTensor::Float32(array.clone()),
        }
    }
}



fn main()-> OrtResult<()> {
    tracing_subscriber::fmt::init();
    // let mut rng = rand::thread_rng();
    let args: Vec<String> = env::args().collect::<Vec<_>>();
    let model_name = &args[1];
    let default_number = String::from("10");
    let number = args.get(2).unwrap_or(&default_number);
    let number = number.parse::<usize>().unwrap();

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
        .with_model_from_file(model_name)?;
    let mut inputs:Vec<InputTensor> = Vec::new();
    // let mut inputs:Vec<Value> = Vec::new();
    // let mut i64_inputs:Vec<Box<CowArray<dyn IntoTensorElementDataType,IxDyn>>> = Vec::new();
    for Input{name,input_type,dimensions}  in session.inputs.iter(){
        let dimensions:Vec<usize> = dimensions.iter().map(|d| return if d.is_some() { d.unwrap() as usize } else { 1 }).collect();
        println!("name: {:?}, input_type: {:?}, dimensions: {:?}",name,input_type,dimensions);
        match input_type {
            ort::tensor::TensorElementDataType::Float32 =>{
                let array = Array::<f32, _>::zeros(dimensions);
                // let array = Rc::new(array);
                // let array = Value::from_array(session.allocator(),&array )?;
                // inputs.push(array);
                inputs.push(InputTensor::Float32(array));
            }
            ort::tensor::TensorElementDataType::Int64 =>{
                let array = Array::<i64, _>::zeros(dimensions);
                // let array = Rc::new(array);
                // let array = Value::from_array(session.allocator(), &array)?;
                inputs.push(InputTensor::Int64(array));
                // inputs.push(array)
            }
            _ => {}
        }

    }
    let mut time_elapsed = Vec::new();
    for _ in 0..number{
        let mut t_inputs: Vec<Value> = Vec::new();
        t_inputs = inputs.iter().map(|input| {
            input.to_tensor(session.allocator())
        }).collect();
        //  inputs.iter().map(|input| {
        //     match input {
        //         InputTensor::Int64(array) => {
        //             let array = CowArray::from(array);
        //             t_inputs.push(Value::from_array(session.allocator(), &array).unwrap());
        //         },
        //         InputTensor::Float32(array) => {
        //             let array = CowArray::from(array);
        //             t_inputs.push(Value::from_array(session.allocator(), &array).unwrap());
        //         },
        //     };
        //
        // });
        // let inputs = for i in inputs{
        //      i.to_tensor(session.allocator())
        // }
        let time_start = std::time::Instant::now();
        let outputs: Vec<Value> = session.run(t_inputs)?;
        println!("!!! BEGIN!");
        let time_end = std::time::Instant::now();
        let time = time_end.duration_since(time_start).as_millis();
        time_elapsed.push(time);
        println!("!!! END!");

        let output = outputs[0].try_extract()? as OrtOwnedTensor<f32, _>;
        println!("{:?}",output.view().shape());
    }
    let time_elapsed = time_elapsed.iter().sum::<u128>() as f64 / time_elapsed.len() as f64;
    println!("time_elapsed: {:?} ms",time_elapsed);
    Ok(())

}
