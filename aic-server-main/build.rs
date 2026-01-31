fn main() -> anyhow::Result<()> {
    println!("cargo::rerun-if-changed=proto/text_embed.proto");
    tonic_prost_build::compile_protos("proto/text_embed.proto")?;
    Ok(())
}
