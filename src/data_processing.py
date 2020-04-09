from src.nli_data_processing import init_glue_model, get_glue_task_dataloader


def init_model(task_name, output_all_layers, num_hidden_layers, config):
    return init_glue_model(task_name, output_all_layers, num_hidden_layers, config)


def get_task_dataloader(task_name, set_name, tokenizer, args, sampler, batch_size=None, knowledge=None, extra_knowledge=None):
    return get_glue_task_dataloader(task_name, set_name, tokenizer, args, sampler, batch_size, knowledge, extra_knowledge)

