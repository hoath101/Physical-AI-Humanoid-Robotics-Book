import Layout from '@theme/Layout';

// Custom layout wrapper
export default function CustomLayout(props) {
  return (
    <Layout {...props}>
      {props.children}
    </Layout>
  );
}