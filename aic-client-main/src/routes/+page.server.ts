import { PUBLIC_BASE_URL, PUBLIC_BASE_STATIC_ASSET_URL } from '$env/static/public';
import type { Keyframe } from '$lib/types';
import type { Actions } from './$types';

type KeyframeSearchResponse = {
	id: number;
	score: number;
};

export const actions: Actions = {
	default: async ({ request, fetch }) => {
		const formData = await request.formData();
		const promptFile = formData.get('prompt')! as File;

		const prompt = await promptFile.text();
		const topK = Number(formData.get('top-k')!);

		const response = await fetch(new URL('/api/v1/vectors/keyframes/searches', PUBLIC_BASE_URL), {
			method: 'post',
			body: JSON.stringify({ prompt, top_k: topK }),
			headers: {
				'content-type': 'application/json'
			}
		});

		if (response.ok) {
			const searchResponses = (await response.json()) as KeyframeSearchResponse[];
			const keyframes = [];
			for (const response of searchResponses) {
				const keyframePromise = fetch(new URL(`/api/v1/keyframes/${response.id}`, PUBLIC_BASE_URL))
					.then((x) => x.json())
					.then((x) => {
						return {
							id: x.id,
							frame_index: x.frame_index,
							frame_timestamp: x.frame_timestamp,
							video_id: x.video_id,
							video_name: x.video_name,
							watch_url: x.watch_url,
							src: new URL(
								`/static/keyframes/${x.video_name}/${x.name}`,
								PUBLIC_BASE_STATIC_ASSET_URL
							).toString()
						} satisfies Keyframe;
					});
				const keyframe: Keyframe = await keyframePromise;
				keyframes.push(keyframe);
			}
			const filename = promptFile.name;
			return { keyframes, filename };
		}
	}
};
