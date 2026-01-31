<script lang="ts">
	import { enhance } from '$app/forms';
	import { Button } from '$lib/components/ui/button';
	import { Card, CardContent } from '$lib/components/ui/card';
	import { Textarea } from '$lib/components/ui/textarea';
	import { Search } from '@lucide/svelte';
	import type { PageProps } from './$types';

	let { form }: PageProps = $props();
</script>

<form
	method="post"
	enctype="multipart/form-data"
	class="flex h-fit w-full max-w-2xl flex-row items-center gap-2"
	use:enhance
>
	<Textarea name="prompt" class="flex-1 resize-none"></Textarea>
	<input name="top-k" value={100} hidden />
	<Button type="submit">
		<Search />
	</Button>
</form>
{#if form && form.keyframes.length !== 0}
	<Card class="grid w-full max-w-7xl flex-1 grid-cols-1">
		<CardContent>
			{#each form.keyframes as keyframe (keyframe.id)}
				<div class="grid grid-cols-3 gap-2">
					<Card>
						<CardContent class="flex items-center">
							<img src={keyframe.src} alt="" class="w-full" />
						</CardContent>
					</Card>
					<Card class="col-span-2">
						<CardContent>
							<p>Video: {keyframe.video_name}</p>
							<p>Index: {keyframe.frame_index}</p>
							<p>Watch URL: <a href={keyframe.watch_url}>link</a></p>
						</CardContent>
					</Card>
				</div>
			{/each}
		</CardContent>
	</Card>
{/if}
