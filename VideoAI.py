import openai
import os
import tempfile
import cv2
import base64
import logging

from openai import OpenAI

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_api_key(api_key_path="api_key.txt"):

    try:
        with open(api_key_path, "r", encoding="utf-8") as f:
            api_key = f.read().strip()
        logging.info(f"API key read from {api_key_path}")
        return api_key
    except FileNotFoundError:
        logging.error(f"Ошибка: Файл с API-ключом не найден: {api_key_path}")
        return None
    except Exception as e:
        logging.exception(
            f"Ошибка при чтении файла с API-ключом: {e}"
        )
        return None


def analyze_video_frame(video_path, frame_number):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Не удалось открыть видеофайл: {video_path}")
            return None

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, frame = cap.read()
        if not ret:
            logging.warning(
                f"Не удалось прочитать кадр {frame_number} из видео."
            )
            cap.release()
            return None

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_frame:
            frame_path = temp_frame.name
            cv2.imwrite(frame_path, frame)
            logging.debug(f"Frame {frame_number} saved to {frame_path}")

        cap.release()
        return frame_path

    except Exception as e:
        logging.exception(f"Ошибка при извлечении кадра: {e}")
        return None


def get_completion_from_chatgpt(prompt, api_key):
    try:
        client = OpenAI(api_key=api_key)
        logging.debug(f"Connecting to OpenAI with api_key")


        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                },
            ],
            max_tokens=1024,
            temperature=0.7,
        )
        logging.info(f"ChatGPT response: {response.choices[0].message.content}")
        return response.choices[0].message.content
    except Exception as e:
        logging.exception(
            f"Ошибка при взаимодействии с ChatGPT: {e}"
        )
        return None

def get_completion_from_chatgpt_image(prompt, image_path, api_key):
    try:
        client = OpenAI(api_key=api_key)
        logging.debug(f"Connecting to OpenAI with api_key")

        with open(image_path, "rb") as image_file:
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64.b64encode(image_file.read()).decode('utf-8')}",
                                    "detail": "high",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=1024,
                temperature=0.7,
            )
        logging.info(f"ChatGPT response: {response.choices[0].message.content}")
        return response.choices[0].message.content
    except Exception as e:
        logging.exception(
            f"Ошибка при взаимодействии с ChatGPT Vision API: {e}"
        )
        return None


def main(video_path, prompt_path, api_key_path="api_key.txt", output_path="output.txt", frames = 1): #измените frames на меньший при длительном видеоролике
    logging.info("Starting main function")
    api_key = get_api_key(api_key_path)
    if api_key is None:
        logging.error("API-ключ не найден.  Работа программы будет прекращена.")
        return

    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt = f.read()
        logging.info(f"Prompt read from {prompt_path}")
    except FileNotFoundError:
        logging.error(f"Ошибка: Файл промпта не найден: {prompt_path}")
        return
    except Exception as e:
        logging.exception(f"Ошибка при чтении файла промпта: {e}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Не удалось открыть видеофайл: {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()

    logging.info(
        f"Video information: FPS={fps}, Total Frames={total_frames}, Duration={duration}"
    )

    num_frames_to_analyze = int(duration * frames)
    if num_frames_to_analyze > total_frames:
        logging.warning(
            "Количество кадров для анализа превышает общее количество кадров в видео.  Будет проанализирован каждый кадр."
        )
        frame_interval = 1
        num_frames_to_analyze = total_frames
    else:
        frame_interval = total_frames // num_frames_to_analyze

    logging.info(
        f"Analyzing {num_frames_to_analyze} frames, Frame Interval={frame_interval}"
    )

    frame_analysis_results = []
    for i in range(num_frames_to_analyze):
        frame_number = i * frame_interval
        image_path = analyze_video_frame(video_path, frame_number)
        if image_path:
            frame_prompt = (
                f"Кадр {i+1}/{num_frames_to_analyze}. {prompt}"
            )
            logging.info(
                f"Analyzing frame {frame_number}, Prompt: {frame_prompt[:100]}..."
            )
            response = get_completion_from_chatgpt_image(
                frame_prompt, image_path, api_key
            )
            if response:
                frame_analysis_results.append(response)
                logging.info(
                    f"Analysis result for frame {frame_number} added."
                )
            try:
                os.remove(image_path)
                logging.debug(f"Temporary file {image_path} removed")
            except OSError as e:
                logging.exception(
                    f"Не удалось удалить временный файл {image_path}: {e}"
                )
        else:
            logging.warning(f"Не удалось обработать кадр {frame_number}")

    if frame_analysis_results:
        summary_prompt = (
            "Объедини результаты анализа следующих кадров в общий вывод, указав на основные ошибки и общие закономерности в технике спортсмена:\n"
            + "\n".join(frame_analysis_results)
        )

        logging.info(f"Creating summary with prompt: {summary_prompt[:100]}...")

        final_analysis = get_completion_from_chatgpt(
            summary_prompt, api_key
        )
        if final_analysis is None:
            final_analysis = "Не удалось сгенерировать финальный анализ."
            logging.warning("Failed to generate final analysis")
    else:
        final_analysis = "Не удалось проанализировать ни один кадр."
        logging.warning("Failed to analyze any frames")

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_analysis)
        logging.info(f"Ответ записан в файл: {output_path}")
    except Exception as e:
        logging.exception(f"Ошибка при записи в файл: {e}")

    logging.info("Ending main function")


if __name__ == "__main__":
    video_file = str(input("Введите полное название файла, пример ввода: 1.mp4\n"))
    prompt_file = "prompt.txt"
    api_key_file = "api_key.txt"
    output_file = "output.txt"
    main(video_file, prompt_file, api_key_file, output_file)