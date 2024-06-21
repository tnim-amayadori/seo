from config import secrets
from datetime import timedelta
from generative import anticipate_cost, cluster, common_util, get_intent, vectorize
import logging
import time

_path_data = 'data'
_input_path = 'data/input_sample.csv'


def _main(input_path=_input_path):
    # Initialize.
    common_util.initialize_logging(_path_data)

    try:
        logging.info("Starting process.")

        secrets.set_api_key()

        # API利用料を予測
        out_cost = common_util.get_daily_path(get_intent.out_cost)
        total_usd, total_jpy = get_intent.pre_anticipate(input_path, out_cost)

        out_cost = common_util.get_daily_path(vectorize.out_cost_pre)
        tmp_usd, tmp_jpy = vectorize.pre_anticipate(input_path, out_cost)
        total_usd += tmp_usd
        total_jpy += tmp_jpy

        out_cost = common_util.get_daily_path(cluster.out_cost_pre)
        tmp_usd, tmp_jpy = cluster.pre_anticipate(input_path, out_cost)
        total_usd += tmp_usd
        total_jpy += tmp_jpy

        msg = anticipate_cost.print_cost(total_usd, total_jpy, pre_msg="Total API Cost [pre]")

        # 処理実行の確認
        user_input = input(f"{msg} \nDo you want to proceed? (y/n) [default is n=No]: ").strip().lower()
        if user_input != 'y':
            logging.info("Process aborted by user")
            return

        # 各モジュールの処理実行
        logging.info('Start to get intents.')
        start_time = time.time()
        intent_path = common_util.get_daily_path(get_intent.out_intent)
        get_intent.main(input_path, intent_path)
        elapsed_time = time.time() - start_time
        formatted_time = str(timedelta(seconds=elapsed_time))
        logging.info(f"Getting intents executed in {formatted_time} seconds.")

        logging.info('Start to vectorize.')
        start_time = time.time()
        np_path = common_util.get_daily_path(vectorize.out_vector)
        vectorize_df_path = common_util.get_daily_path(vectorize.out_vectorize_df)
        cost_path = common_util.get_daily_path(vectorize.out_cost_name)
        vectorize.main(intent_path, np_path, vectorize_df_path, cost_path)
        elapsed_time = time.time() - start_time
        formatted_time = str(timedelta(seconds=elapsed_time))
        logging.info(f"Vector module executed in {formatted_time} seconds.")

        logging.info('Start to cluster.')
        start_time = time.time()
        cost_path = common_util.get_daily_path(cluster.out_cost_name)
        cluster_path = common_util.get_daily_path(cluster.out_cluster_name)
        final_path = common_util.get_daily_path(cluster.out_final_name)
        cluster.main(np_path, vectorize_df_path, cost_path, cluster_path, final_path, input_path)
        elapsed_time = time.time() - start_time
        formatted_time = str(timedelta(seconds=elapsed_time))
        logging.info(f"Clustering executed in {formatted_time} seconds.")

        logging.info("All modules executed successfully.")

    except Exception as e:
        logging.exception(f"An error occurred: {e}")

    finally:
        logging.info("Process completed.")


if __name__ == "__main__":
    _main()
