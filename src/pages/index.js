import Layout from "@theme/Layout";
import Link from "@docusaurus/Link";

export default function Home() {
  return (
    <Layout
      title="Physical AI & Humanoid Robotics"
      description="Bridging Digital Intelligence and Physical Robotic Bodies"
    >
      <main
        style={{
          padding: "80px 20px",
          textAlign: "center",
          maxWidth: "800px",
          margin: "0 auto",
        }}
      >
        <h1 style={{ fontSize: "3rem", marginBottom: "1rem" }}>
          Physical AI & Humanoid Robotics
        </h1>

        <p style={{ fontSize: "1.3rem", lineHeight: "1.6", marginBottom: "2rem" }}>
          Explore how artificial intelligence integrates with physical robotic 
          bodies—unlocking movement, perception, learning, and autonomy.  
          This book bridges theory with real-world robotics and the future of humanoid machines.
        </p>

        <Link
          className="button button--primary button--lg"
          to="/docs/intro"
        >
          📘 Start Reading
        </Link>
      </main>
    </Layout>
  );
}
